import os
import json
import math
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as T


class OracleViewDataset(Dataset):
    """
    Dataset for view-level oracle δ supervision.

    One item = one sample (object), containing 12 views.
    """

    def __init__(
        self,
        renders_root: str,
        oracle_json_path: str,
        image_size: int = 224,
        min_valid_combos: int = 8,
    ):
        """
        Args:
            renders_root: path to renders directory
            oracle_json_path: path to oracle_view.json
            image_size: image resize size
            min_valid_combos: minimal successful combos to keep sample
        """
        self.renders_root = renders_root
        self.oracle_json_path = oracle_json_path
        self.min_valid_combos = min_valid_combos

        # Image Preprocessing
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

        with open(self.oracle_json_path, "r") as f:
            self.oracle_data = json.load(f)

        # Build sample index
        self.samples = self._build_index()

    def _build_index(self) -> List[Dict]:
        """
        Scan renders directory and match with oracle_view.json
        """
        samples = []

        for class_name in sorted(os.listdir(self.renders_root)):
            class_dir = os.path.join(self.renders_root, class_name)
            if not os.path.isdir(class_dir):
                continue

            for sample_id in sorted(os.listdir(class_dir)):
                sample_dir = os.path.join(class_dir, sample_id)
                if not os.path.isdir(sample_dir):
                    continue

                if sample_id not in self.oracle_data:
                    continue

                oracle_entry = self.oracle_data[sample_id]
                if oracle_entry["num_combos_success"] < self.min_valid_combos:
                    continue

                cam_path = os.path.join(sample_dir, "camera.json")
                if not os.path.exists(cam_path):
                    continue

                samples.append({
                    "sample_id": sample_id,
                    "sample_dir": sample_dir,
                })

        print(f"[OracleViewDataset] Loaded {len(samples)} valid samples.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.samples[idx]
        sample_id = entry["sample_id"]
        sample_dir = entry["sample_dir"]

        # ---------- load camera ----------
        cam_path = os.path.join(sample_dir, "camera.json")
        with open(cam_path, "r") as f:
            cam_data = json.load(f)

        # Establish mapping view_id -> pose
        pose_map = {}
        for item in cam_data:
            vid = item["view_id"]
            pose_map[f"{vid:03d}.png"] = (
                item["az"], item["el"], item["roll"]
            )

        # mian pose
        main_pose = pose_map["000.png"]

        images = []
        rel_poses = []
        deltas = []
        view_names = []

        oracle_views = self.oracle_data[sample_id]["view"]

        for view_name in sorted(pose_map.keys()):
            img_path = os.path.join(sample_dir, view_name)
            if not os.path.exists(img_path):
                raise FileNotFoundError(img_path)

            # ---------- image ----------
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] Failed to load image: {img_path}")
                print(f'       Error: {e}')
                # Replace with the main view or the zeroed drawing
                img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

            img = self.transform(img)
            images.append(img)

            # ---------- relative pose ----------
            rel_pose = self._compute_relative_pose(
                pose_map[view_name], main_pose
            )
            rel_poses.append(rel_pose)

            # ---------- oracle delta ----------
            if view_name not in oracle_views:
                raise KeyError(f"{view_name} not in oracle for {sample_id}")

            delta_norm = oracle_views[view_name]["delta_norm"]
            deltas.append(delta_norm)

            view_names.append(view_name)

        return {
            "sample_id": sample_id,
            "images": torch.stack(images, dim=0),        # [12, 3, H, W]
            "rel_pose": torch.stack(rel_poses, dim=0),   # [12, 6]
            "delta": torch.tensor(deltas, dtype=torch.float32),  # [12]
            "view_names": view_names,
        }

    @staticmethod
    def _compute_relative_pose(pose, main_pose):
        """
        Compute relative pose and encode with sin/cos.

        pose, main_pose: (az, el, roll) in degrees
        return: Tensor [6]
        """
        az, el, roll = pose
        az0, el0, roll0 = main_pose

        daz = math.radians(az - az0)
        delv = math.radians(el - el0)
        droll = math.radians(roll - roll0)

        return torch.tensor([
            math.sin(daz), math.cos(daz),
            math.sin(delv), math.cos(delv),
            math.sin(droll), math.cos(droll),
        ], dtype=torch.float32)


class InferenceViewDataset(Dataset):
    """
    Dataset for inference stage.
    Only load images + camera poses.
    """

    def __init__(self, renders_root, image_size=224):
        self.samples = []
        self.renders_root = renders_root

        for category in sorted(os.listdir(renders_root)):
            cat_dir = os.path.join(renders_root, category)
            if not os.path.isdir(cat_dir):
                continue

            for sample_id in sorted(os.listdir(cat_dir)):
                sample_dir = os.path.join(cat_dir, sample_id)
                if not os.path.isdir(sample_dir):
                    continue

                camera_path = os.path.join(sample_dir, "camera.json")
                if not os.path.exists(camera_path):
                    continue

                self.samples.append({
                    "sample_id": sample_id,
                    "sample_dir": sample_dir,
                    "camera_path": camera_path,
                })

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def _load_camera(self, camera_path):
        with open(camera_path, "r") as f:
            cams = json.load(f)

        cams = sorted(cams, key=lambda x: x["view_id"])

        az = torch.tensor([c["az"] for c in cams], dtype=torch.float32)
        el = torch.tensor([c["el"] for c in cams], dtype=torch.float32)
        roll = torch.tensor([c["roll"] for c in cams], dtype=torch.float32)

        az = az - az[0]
        el = el - el[0]
        roll = roll - roll[0]

        az = az * math.pi / 180
        el = el * math.pi / 180
        roll = roll * math.pi / 180

        pose = torch.stack([
            torch.sin(az), torch.cos(az),
            torch.sin(el), torch.cos(el),
            torch.sin(roll), torch.cos(roll),
        ], dim=1)



        return pose

    def __getitem__(self, idx):
        item = self.samples[idx]
        sample_dir = item["sample_dir"]

        images = []
        view_names = []

        for i in range(12):
            name = f"{i:03d}.png"
            img_path = os.path.join(sample_dir, name)

            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)

            images.append(img)
            view_names.append(name)

        images = torch.stack(images, dim=0)      # (V, 3, H, W)
        rel_poses = self._load_camera(item["camera_path"])  # (V, 3)

        return {
            "images": images,
            "rel_poses": rel_poses,
            "view_names": view_names,
            "sample_id": item["sample_id"],
        }
