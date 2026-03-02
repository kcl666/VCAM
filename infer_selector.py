import argparse
import torch

from datasets.oracle_view_dataset import InferenceViewDataset
from models.selector_model import SelectorModel


def main(args):
    device = torch.device(args.device)

    dataset = InferenceViewDataset(
        renders_root=args.renders_root,
    )

    model = SelectorModel(
        args,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        for sample in dataset:
            images = sample["images"].unsqueeze(0).to(device)     # (1, V, 3, H, W)
            rel_poses = sample["rel_poses"].unsqueeze(0).to(device)
            view_names = sample["view_names"]
            sample_id = sample["sample_id"]

            delta_pred = model(images, rel_poses)[0]  # (V,)
            # 屏蔽掉主视图index=0
            delta_pred[0] = -1e9

            scores = delta_pred.cpu()
            sorted_idx = torch.argsort(scores, descending=True)

            topk_views = [view_names[i] for i in sorted_idx[: args.topk]]

            print(f"{sample_id}: {topk_views}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Infer View Selector")

    parser.add_argument("--renders_root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)

    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--fusion_mode", type=str, default="add")

    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args)
