import os
import json
import torch
from tqdm import tqdm
from pytorch3d.io import load_objs_as_meshes, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_mesh(path):
    """Load .obj or .ply to PyTorch3D mesh"""
    try:
        if path.endswith(".obj"):
            mesh = load_objs_as_meshes([path], device=device)
            return mesh
        elif path.endswith(".ply"):
            verts, faces = load_ply(path)

            # Handle the case where faces might be a dict
            if isinstance(faces, dict):
                faces = faces.get("vertex_indices", None)

            if faces is None:
                return None

            verts = verts.to(device)
            faces = faces.long().to(device)

            if verts.numel() == 0 or faces.numel() == 0:
                return None
            mesh = Meshes(
                verts=[verts],
                faces=[faces]
            )
            return mesh
        else:
            return None
    except Exception as e:
        print(f"[Mesh Load Error] {path}: {e}]")
        return None

def compute_chamfer(mesh_pred, mesh_gt,num_sample=50000):
    """Given two meshes, compute Chamfer Distance"""
    try:
        # Sample point cloud (B, N, 3) from mesh
        pts_pred = sample_points_from_meshes(mesh_pred, num_sample)
        pts_gt = sample_points_from_meshes(mesh_gt, num_sample)

        loss, _ = chamfer_distance(pts_pred, pts_gt)

        return float(loss.detach().cpu().item())
    except Exception as e:
        print(f"[Chamfer Distance Error] {e}")
        return None


def load_used_views(camera_json_path):
    """
    Parse the 5 candidate view IDs used in the combo from camera.json.
    There are 8 images in stage1_8: 0.png is the main view, and among the remaining 7 images, there are actually 5 real
    candidates + 2 duplicates.
    We only keep the five real candidate perspectives.
    """
    if not os.path.exists(camera_json_path):
        return []

    with open(camera_json_path, "r") as f:
        data = json.load(f)

    used_views = set()

    for item in data:
        if "orig_name" in item:
            used_views.add(item["orig_name"])

    return sorted(list(used_views))


def generate_oracle_raw(combos_small_root, out_path):
    """
    Main function, iterates through all samples under combos_small_complete, generating oracle_raw.json.
    """
    samples = sorted(os.listdir(combos_small_root))
    results = {}

    for sample in tqdm(samples, desc="samples"):
        sample_dir = os.path.join(combos_small_root, sample)
        if not os.path.isdir(sample_dir):
            continue

        # GT model path
        gt_path = os.path.join(sample_dir, "model_normalized.obj")
        if not os.path.exists(gt_path):
            print(f"[Skip] GT missing for sample {sample}")
            continue

        mesh_gt = load_mesh(gt_path)
        if mesh_gt is None:
            print(f"[Skip] Cannot load GT mesh for sample {sample}")
            continue

        combos = []
        combos_dir = sorted([d for d in os.listdir(sample_dir) if d.startswith("combo_")])

        for combo in tqdm(combos_dir, desc=f"{sample} combos", leave=False):
            combo_path = os.path.join(sample_dir, combo)
            mesh_path = os.path.join(combo_path, "mesh.ply")
            camera_json = os.path.join(combo_path, "stage1_8", "camera.json")


            if not os.path.exists(mesh_path):
                print("The combined mesh.ply does not exist, skipping")
                continue

            if not os.path.exists(camera_json):
                print("camera.json does not exist, skipping")
                continue

            # load pred mesh
            mesh_pred = load_mesh(mesh_path)
            if mesh_pred is None:
                print("mesh_pred Load failed, skipping")
                continue

            # compute Chamfer δ
            delta = compute_chamfer(mesh_pred, mesh_gt)
            if delta is None:
                print("delta Calculation failed, skipping")
                continue

            # parse used_views
            used_views = load_used_views(camera_json)
            combos.append({
                "combo_id": combo,
                "used_views": used_views,
                "delta": delta
            })
        print("combos: ", combos)
        # filter: at least 8 combos to be valid
        if len(combos) < 8:
            print(f"[Skip sample] {sample} only {len(combos)} valid combos (<8)")
            continue

        results[sample] = {
            "gt_path": gt_path,
            "num_combos_success": len(combos),
            "combos": combos
        }

    # write out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n[Done] oracle_raw.json generated:", out_path)


# ====================
#       RUN
# ====================
if __name__ == "__main__":
    combos_small_root = f"/data2/zcx/kcl/test/data/combos_small"
    out_path = f"/data2/zcx/kcl/test/data/oracle_raw.json"

    generate_oracle_raw(combos_small_root, out_path)
