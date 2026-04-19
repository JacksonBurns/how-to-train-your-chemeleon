import torch
from chemprop.nn.metrics import LossFunctionRegistry, MetricRegistry, MSE

from .config import DROPOUT_FRACTION

@LossFunctionRegistry.register("rdmse")
@MetricRegistry.register("rdmse")
class RandomDropoutMSE(MSE):
    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        lt_mask: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> None:
        # overrides parent to generate a randomly initialized mask
        random_mask = (torch.rand_like(targets) < DROPOUT_FRACTION).bool()
        mask = random_mask if mask is None else torch.logical_or(random_mask, mask)  # i.e., ignore if either mask requests so
        super().update(preds, targets, mask, weights, lt_mask, gt_mask)
