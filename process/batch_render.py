import os
import json
import random
import subprocess
import math
from pathlib import Path
from tqdm import tqdm
from candidate_views import *
import shutil
import os

# ---------------- Configuration ----------------
BLENDER_PATH = "XXXXX/blender-4.4.0-linux-x64/blender"  # ✅ Blender executable file path
SCRIPT_PATH = "XXXXX/render_script.py"  # ✅ The relative path of render_script.py
DATASET_ROOT = "XXXXX/ShapeNet_Subset_Large/"   # ✅ Model root directory
OUTPUT_ROOT = "XXXXX/renders_large"   # ✅ Target root directory for output images
NUM_VIEWS = 12 # 1 Front + 11 Random
CANDIDATE_ANGLES = generate_candidate_views()


def gen_camera_json(output_dir: Path, views= CANDIDATE_ANGLES):
    path = output_dir/"camera.json"
    with open(path,"w") as f: json.dump(views,f,indent=2)
    return path


def find_models(dataset_root):
    """Find all model paths that contain models/model_normalized.obj"""
    model_paths = []
    for category in os.listdir(dataset_root):
        cat_path = os.path.join(dataset_root, category)
        if not os.path.isdir(cat_path):
            continue
        for model_id in os.listdir(cat_path):
            model_dir = os.path.join(cat_path, model_id)
            obj_path = os.path.join(model_dir, "models", "model_normalized.obj")
            if os.path.exists(obj_path):
                model_paths.append((category, model_id, model_dir, obj_path))
    return model_paths


def render_model(category, model_id, model_dir, obj_path):
    out = Path(OUTPUT_ROOT)/category/model_id
    out.mkdir(parents=True, exist_ok=True)
    cam_json = gen_camera_json(out)
    cmd = [
        BLENDER_PATH, "-b", "-P", SCRIPT_PATH, "--",
        "--model_dir", str(model_dir),
        "--output_dir", str(out),
        "--camera_json", str(cam_json)
    ]
    try:
        env = os.environ.copy()
        env["DISPLAY"] = ":0"
        subprocess.run(cmd, check=True, env=env)
        # After the rendering is successful, copy the .obj file to the output directory
        dest_obj = out / "model_normalized.obj"
        shutil.copy2(obj_path, dest_obj)
        print(f"✅ Copied .obj to {dest_obj}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed render {category}/{model_id}")
    except Exception as e:
        print(f"❌ Failed copy .obj  {category}/{model_id}：{str(e)}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed render {category}/{model_id}")


if __name__ == "__main__":
    models = find_models(DATASET_ROOT)
    print(f"A total of {len(models)} models, each model renders {NUM_VIEWS} views")

    for category, model_id, model_dir, obj_path in tqdm(models):
        render_model(category, model_id, model_dir, obj_path)