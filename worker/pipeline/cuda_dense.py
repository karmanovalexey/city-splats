"""CUDA depth alignment, consistency fusion and stable voxel reduction.

Camera/track metadata and image files are decoded on the host. Numerical arrays stay
on the GPU through depth fitting, projection, consistency checks and voxel sorting.
"""

from __future__ import annotations

import time
import numpy as np
import cupy as cp
from PIL import Image

from dataclasses import dataclass


@dataclass
class _Frame:
    image: object
    camera: object
    depth: cp.ndarray
    scale: tuple[float, float]
    fit_error: float


def _intrinsics(camera):
    name = camera.model_name
    p = camera.params
    if name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        return name, (p[0], p[0], p[1], p[2]), p[3:]
    if name in ("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
        return name, tuple(p[:4]), p[4:]
    raise ValueError(f"CUDA projection has no implementation for {name}")


def _distort(x, y, model, coefficients):
    r2 = x*x + y*y
    if model in ("PINHOLE", "SIMPLE_PINHOLE"):
        return x, y
    if model == "OPENCV_FISHEYE":
        r = cp.sqrt(r2)
        theta = cp.arctan(r)
        t2 = theta*theta
        k1, k2, k3, k4 = coefficients
        scale = cp.where(r > 1e-12, theta*(1+t2*(k1+t2*(k2+t2*(k3+t2*k4))))/cp.maximum(r,1e-12), 1)
        return x*scale, y*scale
    if model in ("SIMPLE_RADIAL", "RADIAL"):
        scale = 1 + coefficients[0]*r2
        if model == "RADIAL":
            scale += coefficients[1]*r2*r2
        return x*scale, y*scale
    k1, k2, p1, p2 = coefficients[:4]
    radial = 1 + k1*r2 + k2*r2*r2
    if model == "FULL_OPENCV":
        k3, k4, k5, k6 = coefficients[4:]
        radial = (radial+k3*r2**3)/(1+k4*r2+k5*r2*r2+k6*r2**3)
    return x*radial+2*p1*x*y+p2*(r2+2*x*x), y*radial+p1*(r2+2*y*y)+2*p2*x*y


def _rays(u, v, spec):
    model, (fx, fy, cx, cy), coefficients = spec
    tx, ty = (u-cx)/fx, (v-cy)/fy
    x, y = tx.copy(), ty.copy()
    if model in ("PINHOLE", "SIMPLE_PINHOLE"):
        return cp.stack((x,y),1)
    # Newton inversion matches COLMAP's distorted-camera unprojection.
    for _ in range(12):
        a,b = _distort(x,y,model,coefficients)
        ax,bx = _distort(x+1e-6,y,model,coefficients)
        ay,by = _distort(x,y+1e-6,model,coefficients)
        j00,j10,j01,j11 = (ax-a)/1e-6,(bx-b)/1e-6,(ay-a)/1e-6,(by-b)/1e-6
        det = j00*j11-j01*j10
        ex,ey = a-tx,b-ty
        x -= (j11*ex-j01*ey)/det
        y -= (-j10*ex+j00*ey)/det
    return cp.stack((x,y),1)


def _sample(depth, u, v):
    h,w = depth.shape
    valid = cp.isfinite(u)&cp.isfinite(v)&(u>=0)&(v>=0)&(u<w)&(v<h)
    iu = cp.nan_to_num(u).astype(cp.int64).clip(0,w-1)
    iv = cp.nan_to_num(v).astype(cp.int64).clip(0,h-1)
    return cp.where(valid,depth[iv,iu],cp.nan)


def voxel_downsample(xyz, rgb, voxel):
    """Keep the first input point in each cell, including parent-before-new priority."""
    xyz, rgb = cp.asarray(xyz), cp.asarray(rgb)
    if not len(xyz):
        return cp.asnumpy(xyz), cp.asnumpy(rgb)
    cells = cp.floor(xyz/voxel).astype(cp.int64)
    cells -= cells.min(0)
    size = cells.max(0)+1
    if float(cp.prod(size.astype(cp.float64))) >= 2**62:
        raise ValueError("voxel key exceeds int64; increase voxel size")
    key = (cells[:,0]*size[1]+cells[:,1])*size[2]+cells[:,2]
    _, first = cp.unique(key,return_index=True)
    return cp.asnumpy(xyz[first]),cp.asnumpy(rgb[first])


def align_frames(rec, names, depth_dir, stride=2, max_fit_error=.2):
    """Fit GPU depth maps; preserve upstream's streaming frame/sink interface."""
    frames = {}
    registered = {im.name:im for im in rec.images.values() if im.has_pose}
    for index,name in enumerate(names):
        image = registered.get(name)
        path = depth_dir/f"{name}.npy"
        if image is None or not path.exists():
            continue
        camera = rec.cameras[image.camera_id]
        disparity = np.load(path)
        h,w = disparity.shape
        sx,sy = camera.width/w,camera.height/h
        observations = [p for p in image.points2D if p.has_point3D()]
        if len(observations)<50:
            continue
        xy = cp.asarray(np.array([p.xy for p in observations]))
        xyz = cp.asarray(np.array([rec.points3D[p.point3D_id].xyz for p in observations]))
        matrix = cp.asarray(image.cam_from_world().matrix())
        z = (xyz@matrix[:,:3].T+matrix[:,3])[:,2]
        disparity_gpu = cp.asarray(disparity)
        d = _sample(disparity_gpu,xy[:,0]/sx,xy[:,1]/sy)
        valid = cp.isfinite(d)&(z>0)
        d,depth_samples = d[valid],z[valid]
        if len(d)<50:
            continue
        inv_z = 1/depth_samples
        keep = cp.ones(len(d),dtype=bool)
        for _ in range(4):
            if int(keep.sum())<2:
                break
            a,b = cp.linalg.lstsq(cp.stack((d[keep],cp.ones(int(keep.sum()))),1),inv_z[keep],rcond=None)[0]
            residual = cp.abs(a*d+b-inv_z)
            mad = cp.median(residual[keep])+1e-12
            keep = residual<3*1.4826*mad
        if float(a)<=0 or not bool(keep.any()):
            continue
        error = float(cp.median(cp.abs(1/cp.maximum(a*d[keep]+b,1e-6)-depth_samples[keep])/depth_samples[keep]))
        if error>max_fit_error:
            continue
        inv_depth = (a*disparity_gpu+b).astype(cp.float32)
        depth = cp.where(inv_depth>1e-6,1/cp.maximum(inv_depth,1e-6),cp.nan).astype(cp.float32)
        frames[index] = _Frame(image, camera, depth, (sx,sy), error)

    if not frames:
        raise RuntimeError("no frame's depth could be fitted to SfM points")
    footprint = float(np.median([float(cp.nanmedian(f.depth))*stride*f.scale[0]/f.camera.mean_focal_length()
                                 for f in frames.values()]))
    return frames, footprint


def fuse_frames(frames, image_root, sink, stride=2, neighbours=(-4,-2,-1,1,2,4),
                min_agree=2, tolerance=.06, workers=6, on_progress=None):
    """Stream consistent GPU-fused points to upstream's tile spill, one frame at a time."""
    ray_cache, count = {}, 0
    for done,(index,frame) in enumerate(frames.items(),1):
        depth=frame.depth; sx,sy=frame.scale; h,w=depth.shape
        spec=_intrinsics(frame.camera)
        camera_key=(frame.camera.camera_id,h,w,stride)
        if camera_key not in ray_cache:
            v,u=cp.mgrid[0:h:stride,0:w:stride];u,v=u.ravel(),v.ravel()
            ray_cache[camera_key]=(u,v,_rays((u+.5)*sx,(v+.5)*sy,spec))
        u,v,all_rays=ray_cache[camera_key]
        z=depth[v,u];valid=cp.isfinite(z)
        u,v,z=u[valid],v[valid],z[valid];rays=all_rays[valid]
        cam=cp.concatenate((rays*z[:,None],z[:,None]),1)
        m=cp.asarray(frame.image.cam_from_world().matrix())
        world=(cam-m[:,3])@m[:,:3]
        agree=cp.zeros(len(world),dtype=cp.int32)
        for offset in neighbours:
            other=frames.get(index+offset)
            if other is None:continue
            m=cp.asarray(other.image.cam_from_world().matrix())
            cam=world@m[:,:3].T+m[:,3];front=cam[:,2]>1e-6
            model,(fx,fy,cx,cy),coef=_intrinsics(other.camera)
            x,y=_distort(cam[:,0]/cam[:,2],cam[:,1]/cam[:,2],model,coef)
            measured=_sample(other.depth,cp.where(front,(fx*x+cx)/other.scale[0],cp.nan),
                             cp.where(front,(fy*y+cy)/other.scale[1],cp.nan))
            agree+=cp.abs(cam[:,2]-measured)<tolerance*measured
        keep=agree>=min_agree
        if bool(keep.any()):
            with Image.open(image_root/frame.image.name) as image:
                rgb=cp.asarray(np.asarray(image.convert('RGB').resize((w,h))))
            xyz=cp.asnumpy(world[keep].astype(cp.float32));colors=cp.asnumpy(rgb[v[keep],u[keep]])
            sink(xyz,colors);count+=len(xyz)
        if on_progress:on_progress(done,len(frames))
    if not count:raise RuntimeError('CUDA densification produced no consistent points')
    return count


def write_cloud_splats(path, xyz, rgb, voxel, opacity=.98):
    """Pack the current tile on CUDA; emit the same Gaussian PLY fields as upstream."""
    from .cleanup import write_ply,SH_C0
    n=len(xyz);rows=cp.zeros((n,14),dtype=cp.float32)
    rows[:,:3]=cp.asarray(xyz);rows[:,3:6]=np.log(.6*voxel)
    rows[:,6]=np.log(opacity/(1-opacity));rows[:,7]=1
    rows[:,11:14]=(cp.asarray(rgb,dtype=cp.float32)/255-.5)/SH_C0
    props=['x','y','z','scale_0','scale_1','scale_2','opacity','rot_0','rot_1','rot_2','rot_3','f_dc_0','f_dc_1','f_dc_2']
    header=f'ply\nformat binary_little_endian 1.0\nelement vertex {n}\n'+''.join(f'property float {p}\n' for p in props)+'end_header\n'
    write_ply(path,header,cp.asnumpy(rows))
