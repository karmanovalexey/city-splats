"""Paired SfM solver tests on copies of saved DJI models; never creates a scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pycolmap


def quality(rec):
    errors=[]
    for image in rec.images.values():
        if not image.has_pose:
            continue
        observations=[p for p in image.points2D if p.has_point3D()]
        if not observations:
            continue
        xyz=np.array([rec.points3D[p.point3D_id].xyz for p in observations])
        xy=np.array([p.xy for p in observations])
        m=image.cam_from_world().matrix()
        cam=xyz@m[:,:3].T+m[:,3]
        projected=rec.cameras[image.camera_id].img_from_cam(cam)
        e=np.linalg.norm(projected-xy,axis=1)
        errors.extend(e[np.isfinite(e)&(cam[:,2]>0)])
    e=np.asarray(errors)
    return dict(observations=len(e),mean_reprojection_px=float(e.mean()),
                median_reprojection_px=float(np.median(e)),rms_reprojection_px=float(np.sqrt(np.mean(e*e))),
                p95_reprojection_px=float(np.quantile(e,.95)))


def prepare(source,destination):
    rec=pycolmap.Reconstruction(source)
    centers=np.array([im.projection_center() for im in rec.images.values() if im.has_pose])
    radius=float(np.median(np.linalg.norm(centers-centers.mean(0),axis=1)))
    rng=np.random.default_rng(42)
    for fid in sorted(rec.frames):
        frame=rec.frames[fid]
        if frame.has_pose():
            pose=frame.rig_from_world
            frame.rig_from_world=pycolmap.Rigid3d(pose.rotation,pose.translation+rng.normal(size=3)*radius*.005)
    for pid in sorted(rec.points3D):
        p=rec.points3D[pid]
        p.xyz=p.xyz+rng.normal(size=3)*radius*.001
    destination.mkdir(parents=True,exist_ok=True)
    rec.write(destination)
    digest=hashlib.sha256()
    for p in sorted(destination.glob('*.bin')):
        digest.update(p.name.encode());digest.update(p.read_bytes())
    metadata=dict(source=str(source.resolve()),images=rec.num_reg_images(),points=rec.num_points3D(),
                  seed=42,camera_translation_noise_sigma=radius*.005,point_noise_sigma=radius*.001,
                  input_sha256=digest.hexdigest(),initial_quality=quality(rec))
    (destination/'input.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)


def run(args):
    rec=pycolmap.Reconstruction(args.model)
    gpu=args.backend=='cuda'
    record=dict(stage=args.stage,backend=args.backend,model=str(args.model.resolve()),
                images=rec.num_reg_images(),points=rec.num_points3D(),threads=args.threads,
                input=json.loads((args.model/'input.json').read_text()))
    if args.stage=='ba':
        config=pycolmap.BundleAdjustmentConfig()
        for iid in rec.reg_image_ids():
            config.add_image(iid)
        config.fix_gauge(pycolmap.BundleAdjustmentGauge.THREE_POINTS)
        options=pycolmap.BundleAdjustmentOptions()
        options.print_summary=False
        options.ceres.use_gpu=gpu
        options.ceres.gpu_index='0'
        options.ceres.min_num_images_gpu_solver=0
        options.ceres.auto_select_solver_type=False
        solver_options=options.ceres.solver_options
        solver_options.num_threads=args.threads
        solver_options.max_num_iterations=args.iterations
        solver_options.max_solver_time_in_seconds=120
        solver_options.linear_solver_type='SPARSE_SCHUR'
        solver_options.sparse_linear_algebra_library_type='CUDA_SPARSE' if gpu else 'SUITE_SPARSE'
        start=time.perf_counter()
        adjuster=pycolmap.create_default_ceres_bundle_adjuster(options,config,rec)
        summary=adjuster.solve()
        record['wall_seconds']=time.perf_counter()-start
        s=summary.ceres_summary
        fields=['initial_cost','final_cost','num_successful_steps','num_unsuccessful_steps',
                'total_time_in_seconds','linear_solver_time_in_seconds','num_linear_solves',
                'residual_evaluation_time_in_seconds','jacobian_evaluation_time_in_seconds',
                'preprocessor_time_in_seconds','postprocessor_time_in_seconds','num_threads_used',
                'num_residuals','num_effective_parameters_reduced']
        record.update({field:getattr(s,field) for field in fields})
        record.update(usable=s.IsSolutionUsable(),termination=str(s.termination_type),message=s.message,
                      sparse_library=str(s.sparse_linear_algebra_library_type),
                      linear_solver=str(s.linear_solver_type_used))
        args.output.with_suffix('.report.txt').write_text(s.FullReport())
        if gpu and 'CUDA_SPARSE' not in record['sparse_library']:
            raise RuntimeError('CUDA requested but solver fell back to CPU')
    else:
        options=pycolmap.GlobalPositionerOptions()
        options.use_gpu=gpu
        options.gpu_index='0'
        options.min_num_images_gpu_solver=0
        options.random_seed=42
        options.generate_random_positions=False
        options.generate_random_points=False
        options.use_parameter_block_ordering=args.pipeline_ordering
        # Only existing tracks are needed for this isolated positioning solve.
        graph=pycolmap.PoseGraph()
        start=time.perf_counter()
        record['usable']=pycolmap.run_global_positioning(options,graph,rec)
        record['wall_seconds']=time.perf_counter()-start
        record['initialization']='identical saved perturbed poses/points; rotations fixed'
        record['pipeline_parameter_ordering']=args.pipeline_ordering
        record['threads']='COLMAP default (28 logical CPUs)'
    record['quality']=quality(rec)
    record['camera_centers']={im.name:im.projection_center().tolist() for im in rec.images.values() if im.has_pose}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k!='camera_centers'}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--stage',choices=['ba','position'],default='ba')
    parser.add_argument('--backend',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--iterations',type=int,default=20)
    parser.add_argument('--threads',type=int,default=28)
    parser.add_argument('--pipeline-ordering',action='store_true',help='Use the pipeline default ordering for global positioning')
    args=parser.parse_args()
    if args.prepare:
        prepare(args.prepare,args.model)
    else:
        if not args.output:
            parser.error('--output is required for a solve')
        run(args)


if __name__=='__main__':
    main()
