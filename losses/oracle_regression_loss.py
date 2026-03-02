import torch
import torch.nn as nn
import torch.nn.functional as F


class OracleRegressionLoss(nn.Module):
    """
    Regression loss for oracle view importance (delta).

    Pure view-wise regression loss.
    Optionally ignores the main view (e.g. 000.png).
    """

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        reduction: str = "mean",
        ignore_main_view: bool = False,
        main_view_index: int = 0,
    ):
        super().__init__()

        assert loss_type in ["l1", "mse", "smooth_l1"]
        assert reduction in ["mean", "sum"]

        self.loss_type = loss_type
        self.reduction = reduction
        self.ignore_main_view = ignore_main_view
        self.main_view_index = main_view_index

    def forward(
        self,
        delta_pred: torch.Tensor,
        delta_gt: torch.Tensor,
    ):
        """
        delta_pred: (B, V)
        delta_gt:   (B, V)

        return:
            scalar loss
        """

        assert delta_pred.shape == delta_gt.shape
        B, V = delta_pred.shape

        # ===== optionally ignore main view =====
        if self.ignore_main_view:
            mask = torch.ones_like(delta_gt, dtype=torch.bool)
            mask[:, self.main_view_index] = False

            # 若全部被 mask（理论上不会，但防御一下）
            if mask.sum() == 0:
                return torch.tensor(
                    0.0,
                    device=delta_pred.device,
                    requires_grad=True,
                )

            pred = delta_pred[mask]
            gt = delta_gt[mask]
        else:
            pred = delta_pred
            gt = delta_gt

        # ===== loss computation =====
        if self.loss_type == "l1":
            loss = F.l1_loss(pred, gt, reduction=self.reduction)
        elif self.loss_type == "mse":
            loss = F.mse_loss(pred, gt, reduction=self.reduction)
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(pred, gt, reduction=self.reduction)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss
