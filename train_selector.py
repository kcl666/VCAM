import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.oracle_view_dataset import OracleViewDataset
from models.selector_model import SelectorModel
from losses.oracle_regression_loss import OracleRegressionLoss


def main(args):
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    # ===== Dataset =====
    dataset = OracleViewDataset(
        renders_root=args.renders_root,
        oracle_json_path=args.oracle_json,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ===== Model =====
    model = SelectorModel(
        args,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = OracleRegressionLoss(
        loss_type="smooth_l1",
        ignore_main_view=args.ignore_main_view,
    )

    start_epoch = 0
    global_step = 0

    # ===== Resume checkpoint =====
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        print(f"[Resume] epoch={start_epoch}, global_step={global_step}")

    model.train()

    # ===== Training loop =====
    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch in pbar:
            images = batch["images"].to(device)        # (B, V, 3, H, W)
            rel_poses = batch["rel_pose"].to(device)  # (B, V, 3)
            delta_gt = batch["delta"].to(device)       # (B, V)

            delta_pred = model(images, rel_poses)      # (B, V)

            loss = criterion(
                delta_pred,
                delta_gt,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

            # ===== Step-level checkpoint =====
            if global_step % args.save_every_steps == 0:
                ckpt_path = os.path.join(
                    args.output_dir, f"step_{global_step}.pth"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    ckpt_path,
                )

        # ===== Epoch-level checkpoint =====
        if (epoch + 1) % args.save_every_epochs == 0:
            ckpt_path = os.path.join(
                args.output_dir, f"epoch_{epoch+1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_path,
            )


    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train View Selector")

    # ===== Data =====
    parser.add_argument("--renders_root", type=str, required=True)
    parser.add_argument("--oracle_json", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    # ===== Model =====
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--fusion_mode", type=str, default="add")

    # ===== Training =====
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ignore_main_view", action="store_true")

    # ===== Checkpoint =====
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_every_steps", type=int, default=1000)
    parser.add_argument("--save_every_epochs", type=int, default=5)

    # ===== Device =====
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args)
