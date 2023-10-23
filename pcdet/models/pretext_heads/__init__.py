from .seg_head import SegHead
from .seg_vox_head import SegVoxHead
from .depth_head import DepthHead
from .reg_heads import RegHead



__all__ = {
    'SegHead': SegHead,
    'SegVoxHead': SegVoxHead,
    'DepthHead': DepthHead,
    'RegHead': RegHead
    }
