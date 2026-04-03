from typing import List, Tuple, Optional, Any, Union

from .clip import _clip_ebc, CLIP_EBC
import assets

clip_names = ["resnet50", "resnet50x4", "resnet50x16", "resnet50x64", "resnet101", "vit_b_16", "vit_b_32", "vit_l_14"]


def get_model(
    backbone: str,
    input_size: int,
    reduction: int,
    bins: Optional[List[Tuple[float, float]]] = None,
    anchor_points: Optional[List[float]] = None,
    **kwargs: Any,
) -> CLIP_EBC:
    backbone = backbone.lower()
    if "clip" in backbone:
        backbone = backbone[5:]
    assert backbone in clip_names, f"Expected backbone to be in {clip_names}, got {backbone}"
    return _clip_ebc(
        backbone=backbone,
        input_size=input_size,
        reduction=reduction,
        bins=bins,
        anchor_points=anchor_points,
        **kwargs
    )


__all__ = ["get_model"]
