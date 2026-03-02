import os
import json
import numpy as np
import trimesh
import open3d as o3d
from tqdm import tqdm

# ----------------------------
# Base_utils
# ----------------------------

def load_mesh(path):
    mesh = trimesh.load(path, force='mesh', process=False, skip_materials=True)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(mesh.dump())
    return mesh


def normalize_mesh(mesh):
    v = mesh.vertices
    center = v.mean(axis=0)
    v = v - center
    scale = np.max(np.linalg.norm(v, axis=1))
    if scale < 1e-8:
        scale = 1.0
    v = v / scale
    mesh.vertices = v
    return mesh


def mesh_to_pcd(mesh, n=5000):
    if len(mesh.vertices) < n:
        pts = mesh.vertices
    else:
        pts = mesh.sample(n)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def icp_align(src_mesh, tgt_mesh, n=5000):
    src = mesh_to_pcd(src_mesh, n)
    tgt = mesh_to_pcd(tgt_mesh, n)

    threshold = 0.05
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    T = reg.transformation
    src_mesh.apply_transform(T)
    return src_mesh, reg.fitness


# ----------------------------
# Indicator
# ----------------------------

def chamfer_distance(mesh1, mesh2, n=5000):
    p1 = mesh1.sample(n)
    p2 = mesh2.sample(n)

    kdt1 = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(p1))
    kdt2 = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(p2))

    d1 = []
    for pt in p2:
        _, _, dist = kdt1.search_knn_vector_3d(pt, 1)
        d1.append(dist[0])

    d2 = []
    for pt in p1:
        _, _, dist = kdt2.search_knn_vector_3d(pt, 1)
        d2.append(dist[0])

    return (np.mean(d1) + np.mean(d2)) / 2


def fscore(mesh1, mesh2, tau=0.01, n=5000):
    p1 = mesh1.sample(n)
    p2 = mesh2.sample(n)

    kdt1 = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(p1))
    kdt2 = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(p2))

    tp1 = 0
    for pt in p2:
        _, _, dist = kdt1.search_knn_vector_3d(pt, 1)
        if dist[0] < tau:
            tp1 += 1

    tp2 = 0
    for pt in p1:
        _, _, dist = kdt2.search_knn_vector_3d(pt, 1)
        if dist[0] < tau:
            tp2 += 1

    recall = tp1 / len(p2)
    precision = tp2 / len(p1)

    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def voxel_iou(mesh1, mesh2, res=64):
    v1 = mesh1.vertices
    v2 = mesh2.vertices

    def voxelize(v):
        v = (v + 1) / 2
        v = np.clip(v, 0, 0.9999)
        idx = (v * res).astype(int)
        grid = np.zeros((res, res, res), dtype=bool)
        grid[idx[:,0], idx[:,1], idx[:,2]] = True
        return grid

    g1 = voxelize(v1)
    g2 = voxelize(v2)

    inter = np.logical_and(g1, g2).sum()
    union = np.logical_or(g1, g2).sum()

    if union == 0:
        return 0.0
    return inter / union


# ----------------------------
# Single-sample Evaluation
# ----------------------------

def eval_pair(pred_path, gt_path):
    gt = load_mesh(gt_path)
    pred = load_mesh(pred_path)

    gt = normalize_mesh(gt)
    pred = normalize_mesh(pred)

    pred, fitness = icp_align(pred, gt)

    cd = chamfer_distance(pred, gt)
    f1 = fscore(pred, gt, tau=0.01)
    iou = voxel_iou(pred, gt)

    return {
        "CD": float(cd),
        "F1": float(f1),
        "IoU": float(iou),
        "ICP_fitness": float(fitness)
    }


# ----------------------------
# Batch Evaluation Entry
# ----------------------------

"""
Assumed directory structure:

dataset_root/
  GSO_subset/
    alarm/
      gt.obj
      raw.obj
      opt.obj
    bell/
      ...

Can modify the path concatenation according to your own structure
"""

DATASETS = {
    "GSO_subset": "datasets/GSO_subset",
    "Objaverse_subset": "datasets/Objaverse_subset",
    "ShapeNetCore_subset": "datasets/ShapeNetCore_subset"
}

GT_NAME = "gt.obj"
RAW_NAME = "raw.obj"
OPT_NAME = "opt.obj"


def main():
    results = {}

    for dname, droot in DATASETS.items():
        print(f"\n[INFO] Dataset: {dname}")
        results[dname] = {}

        ids = sorted(os.listdir(droot))

        for sid in tqdm(ids):
            spath = os.path.join(droot, sid)
            if not os.path.isdir(spath):
                continue

            try:
                gt_path = os.path.join(spath, GT_NAME)
                raw_path = os.path.join(spath, RAW_NAME)
                opt_path = os.path.join(spath, OPT_NAME)

                raw_metrics = eval_pair(raw_path, gt_path)
                opt_metrics = eval_pair(opt_path, gt_path)

                results[dname][sid] = {
                    "raw": raw_metrics,
                    "opt": opt_metrics
                }

            except Exception as e:
                print(f"[WARN] {dname}/{sid} failed: {e}")

    with open("eval_results_icp.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[OK] Saved to eval_results_icp.json")


if __name__ == "__main__":
    main()
