from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.mobilenetv3 import InvertedResidual
from torchvision.ops.misc import SqueezeExcitation


@dataclass
class BlockInfo:
    """Holds references to the submodules we need to edit inside an MBConv block."""

    index: int
    block: InvertedResidual
    expand_layer: Optional[nn.Sequential]
    expand_conv: Optional[nn.Conv2d]
    expand_bn: Optional[nn.BatchNorm2d]
    depthwise_layer: nn.Sequential
    depthwise_conv: nn.Conv2d
    depthwise_bn: nn.BatchNorm2d
    project_layer: nn.Sequential
    project_conv: nn.Conv2d
    project_bn: nn.BatchNorm2d
    se_layer: Optional[nn.Module]
    channels: List[int] = field(default_factory=list)
    min_channels: int = 0
    removed_history: List[int] = field(default_factory=list)

    def prune(self, remove_indices: Sequence[int]) -> List[int]:
        """Physically remove channels from this block."""
        unique_indices = sorted(set(int(i) for i in remove_indices))
        if not unique_indices:
            return []

        if self.expand_conv is None or self.expand_bn is None:
            return []

        original_channels = list(self.channels)
        keep_indices = [i for i in range(len(original_channels)) if i not in unique_indices]

        if len(keep_indices) < self.min_channels:
            raise ValueError(
                f"Block {self.index}: pruning would drop below min_channels "
                f"({len(keep_indices)} < {self.min_channels})."
            )

        removed_original = [original_channels[i] for i in unique_indices]

        self._apply_selection(keep_indices)
        self.channels = [original_channels[i] for i in keep_indices]
        self.removed_history.extend(removed_original)

        return removed_original

    def _apply_selection(self, keep_indices: Sequence[int]) -> None:
        keep_list = list(keep_indices)
        device = self.expand_conv.weight.device
        keep_tensor = torch.as_tensor(keep_list, device=device, dtype=torch.long)

        # Expand conv + BN
        self.expand_conv.weight = nn.Parameter(
            torch.index_select(self.expand_conv.weight.detach(), dim=0, index=keep_tensor).clone()
        )
        self.expand_conv.out_channels = len(keep_list)
        if self.expand_conv.bias is not None:
            self.expand_conv.bias = nn.Parameter(
                torch.index_select(self.expand_conv.bias.detach(), dim=0, index=keep_tensor).clone()
            )
        if self.expand_bn is not None:
            self._slice_batch_norm(self.expand_bn, keep_tensor)

        # Depthwise conv + BN
        self.depthwise_conv.weight = nn.Parameter(
            torch.index_select(self.depthwise_conv.weight.detach(), dim=0, index=keep_tensor).clone()
        )
        self.depthwise_conv.in_channels = len(keep_list)
        self.depthwise_conv.out_channels = len(keep_list)
        self.depthwise_conv.groups = len(keep_list)
        if self.depthwise_conv.bias is not None:
            self.depthwise_conv.bias = nn.Parameter(
                torch.index_select(self.depthwise_conv.bias.detach(), dim=0, index=keep_tensor).clone()
            )
        if self.depthwise_bn is not None:
            self._slice_batch_norm(self.depthwise_bn, keep_tensor)

        # Squeeze & Excitation
        if isinstance(self.se_layer, SqueezeExcitation):
            fc1 = self.se_layer.fc1
            fc2 = self.se_layer.fc2
            fc1.weight = nn.Parameter(
                torch.index_select(fc1.weight.detach(), dim=1, index=keep_tensor).clone()
            )
            fc1.in_channels = len(keep_list)
            if fc1.bias is not None:
                fc1.bias = nn.Parameter(fc1.bias.detach().clone())

            fc2.weight = nn.Parameter(
                torch.index_select(fc2.weight.detach(), dim=0, index=keep_tensor).clone()
            )
            if fc2.bias is not None:
                fc2.bias = nn.Parameter(
                    torch.index_select(fc2.bias.detach(), dim=0, index=keep_tensor).clone()
                )
            fc2.out_channels = len(keep_list)

        # Project conv + BN
        input_select = keep_tensor
        self.project_conv.weight = nn.Parameter(
            torch.index_select(self.project_conv.weight.detach(), dim=1, index=input_select).clone()
        )
        self.project_conv.in_channels = len(keep_list)
        if self.project_conv.bias is not None:
            # Bias is aligned with output channels, so no change expected
            self.project_conv.bias = nn.Parameter(self.project_conv.bias.detach().clone())
        if self.project_bn is not None:
            # Project BN stays the same (output channels unaffected)
            pass

    @staticmethod
    def _slice_batch_norm(bn: nn.BatchNorm2d, keep_tensor: Tensor) -> None:
        bn.weight = nn.Parameter(torch.index_select(bn.weight.detach(), 0, keep_tensor).clone())
        bn.bias = nn.Parameter(torch.index_select(bn.bias.detach(), 0, keep_tensor).clone())
        bn.running_mean = torch.index_select(bn.running_mean.detach(), 0, keep_tensor).clone()
        bn.running_var = torch.index_select(bn.running_var.detach(), 0, keep_tensor).clone()
        bn.num_features = keep_tensor.numel()


@dataclass(order=True)
class ChannelScore:
    score: float
    block_index: int
    channel_idx: int
    original_idx: int = field(compare=False)


class MobilenetV3ChannelPruner:
    """Utilities to perform structured channel pruning on torchvision MobileNetV3 blocks."""

    def __init__(
        self,
        model: nn.Module,
        protect_cfg: Optional[Dict] = None,
        importance: str = "bn_gamma",
        expand_only: bool = True,
    ):
        self.model = model
        self.importance = importance
        self.expand_only = expand_only
        self.protect_cfg = protect_cfg or {}
        self.blocks: List[BlockInfo] = self._collect_blocks()
        self.total_initial_prunable = sum(
            max(0, len(block.channels) - block.min_channels) for block in self.blocks
        )
        self.removed_so_far = 0

    def _collect_blocks(self) -> List[BlockInfo]:
        blocks: List[BlockInfo] = []
        block_min_default = int(self.protect_cfg.get("block_min_channels", 16))
        keep_block_indices = set(self.protect_cfg.get("keep_block_indices", []))
        skip_first_n = int(self.protect_cfg.get("skip_first_n", 0))
        per_block_min = self.protect_cfg.get("per_block_min_channels", {})

        for idx, block in enumerate(self.model.features):
            # Skip any block that isn't inverted residual, in the first n blocks, or specifically blocked
            if skip_first_n and idx < skip_first_n:
                continue
            if keep_block_indices and idx in keep_block_indices:
                continue
            if not isinstance(block, InvertedResidual):
                continue

            # Locate each layer inside the block
            expand_layer, expand_conv, expand_bn = self._maybe_expand(block)
            if expand_conv is None or expand_bn is None:
                # Skip blocks without an expand conv when expand_only is requested.
                if self.expand_only:
                    continue
            depthwise_layer, depthwise_conv, depthwise_bn, project_start = self._depthwise(block, expand_layer is not None)
            project_layer, project_conv, project_bn = self._project(block, project_start)
            se_layer = self._maybe_se(block, expand_layer is not None)

            block_min = int(per_block_min.get(idx, block_min_default))
            block_min = max(block_min, 0)

            channels = (
                list(range(expand_conv.out_channels))
                if expand_conv is not None
                else []
            )

            blocks.append(
                BlockInfo(
                    index=idx,
                    block=block,
                    expand_layer=expand_layer,
                    expand_conv=expand_conv,
                    expand_bn=expand_bn,
                    depthwise_layer=depthwise_layer,
                    depthwise_conv=depthwise_conv,
                    depthwise_bn=depthwise_bn,
                    project_layer=project_layer,
                    project_conv=project_conv,
                    project_bn=project_bn,
                    se_layer=se_layer,
                    channels=channels,
                    min_channels=block_min,
                )
            )

        return blocks

    @staticmethod
    def _first_conv(module: nn.Sequential) -> Optional[nn.Conv2d]:
        for sub in module:
            if isinstance(sub, nn.Conv2d):
                return sub
        return None

    def _maybe_expand(
        self, block: InvertedResidual
    ) -> Tuple[Optional[nn.Sequential], Optional[nn.Conv2d], Optional[nn.BatchNorm2d]]:
        """Check if block contains any expand pointwise convs"""
        layers = list(block.block)
        if not layers:
            return None, None, None
        first = layers[0]
        if not isinstance(first, nn.Sequential):
            return None, None, None
        conv = self._first_conv(first)
        if conv is None:
            return None, None, None
        if conv.kernel_size != (1, 1) or conv.groups != 1:
            return None, None, None

        bn = None
        for sub in first:
            if isinstance(sub, nn.BatchNorm2d):
                bn = sub
                break

        if bn is None:
            return None, None, None

        return first, conv, bn

    def _depthwise(
        self, block: InvertedResidual, has_expand: bool
    ) -> Tuple[nn.Sequential, nn.Conv2d, nn.BatchNorm2d, int]:
        """Check if block contains depthwise conv"""
        layers = list(block.block)
        idx = 1 if has_expand else 0
        depthwise_layer = layers[idx]
        conv = self._first_conv(depthwise_layer)
        if conv is None:
            raise RuntimeError("Expected depthwise conv inside InvertedResidual.")
        bn = None
        for sub in depthwise_layer:
            if isinstance(sub, nn.BatchNorm2d):
                bn = sub
                break
        if bn is None:
            raise RuntimeError("Expected BatchNorm after depthwise conv.")
        return depthwise_layer, conv, bn, idx + 1

    def _maybe_se(self, block: InvertedResidual, has_expand: bool) -> Optional[nn.Module]:
        """Check if block contains squeeze excitation layer"""
        layers = list(block.block)
        idx = 2 if has_expand else 1
        if idx >= len(layers):
            return None
        candidate = layers[idx]
        if isinstance(candidate, SqueezeExcitation):
            return candidate
        return None

    def _project(
        self,
        block: InvertedResidual,
        start_idx: int,
    ) -> Tuple[nn.Sequential, nn.Conv2d, nn.BatchNorm2d]:
        """Check if block contains project pointwise conv"""
        layers = list(block.block)
        if start_idx >= len(layers):
            raise RuntimeError("Project conv missing in InvertedResidual.")

        candidate = layers[start_idx]
        # In configurations with SE, the project layer is at start_idx or start_idx + 1
        if isinstance(candidate, SqueezeExcitation):
            start_idx += 1
            candidate = layers[start_idx]

        project_layer = candidate
        conv = self._first_conv(project_layer)
        if conv is None:
            raise RuntimeError("Expected project conv inside InvertedResidual.")
        bn = None
        for sub in project_layer:
            if isinstance(sub, nn.BatchNorm2d):
                bn = sub
                break
        if bn is None:
            raise RuntimeError("Expected BatchNorm after project conv.")
        return project_layer, conv, bn

    def available_prunable_channels(self) -> int:
        return sum(max(0, len(block.channels) - block.min_channels) for block in self.blocks)

    def current_fraction(self) -> float:
        if self.total_initial_prunable <= 0:
            return 0.0
        return self.removed_so_far / float(self.total_initial_prunable)

    def compute_scores(self, importance: str) -> List[ChannelScore]:
        """Compute importance scores for all prunable channels."""
        metric = (importance or self.importance).lower()
        scores: List[ChannelScore] = []
        for block in self.blocks:
            if not block.channels:
                continue
            if len(block.channels) <= block.min_channels:
                continue

            if metric == "bn_gamma" and block.expand_bn is not None:
                values = block.expand_bn.weight.detach().abs().cpu()
            elif metric == "l1_norm":
                weight = block.expand_conv.weight.detach().abs()
                values = weight.view(weight.shape[0], -1).mean(dim=1).cpu()
            else:
                raise ValueError(f"Invalid importance type: {metric}")

            for channel_idx, original_idx in enumerate(block.channels):
                scores.append(
                    ChannelScore(
                        score=float(values[channel_idx].item()),
                        block_index=block.index,
                        channel_idx=channel_idx,
                        original_idx=original_idx,
                    )
                )
        scores.sort()
        return scores

    def select_channels(self, target_remove: int, importance: str) -> List[ChannelScore]:
        """Select channels to prune based on importance scores."""
        if target_remove <= 0:
            return []

        scores = self.compute_scores(importance)
        chosen: List[ChannelScore] = []
        allocated: Dict[int, int] = defaultdict(int)

        for score in scores:
            block = self._block_by_index(score.block_index)
            if block is None:
                continue
            current = len(block.channels) - allocated[block.index]
            if current - 1 < block.min_channels:
                continue
            chosen.append(score)
            allocated[block.index] += 1
            if len(chosen) >= target_remove:
                break

        return chosen

    def apply(self, plan: Sequence[ChannelScore]) -> Dict[int, Dict[str, object]]:
        """Apply pruning plan to the model."""
        per_block: Dict[int, Dict[str, object]] = {}
        grouped: Dict[int, List[int]] = defaultdict(list)

        for score in plan:
            grouped[score.block_index].append(score.channel_idx)

        total_removed = 0
        for block_idx, indices in grouped.items():
            block = self._block_by_index(block_idx)
            if block is None:
                continue
            removed = block.prune(indices)
            total_removed += len(removed)
            per_block[block_idx] = {
                "removed_channels": removed,
                "remaining": len(block.channels),
            }

        self.removed_so_far += total_removed
        return per_block

    def _block_by_index(self, block_index: int) -> Optional[BlockInfo]:
        for block in self.blocks:
            if block.index == block_index:
                return block
        return None

    def plan_for_fraction(self, target_fraction: float, importance: str) -> List[ChannelScore]:
        """Create a pruning plan to reach the target fraction of pruned channels."""
        fraction = max(0.0, min(1.0, float(target_fraction)))
        target_total = int(round(self.total_initial_prunable * fraction))
        to_remove = target_total - self.removed_so_far
        if to_remove <= 0:
            return []
        return self.select_channels(to_remove, importance=importance)
