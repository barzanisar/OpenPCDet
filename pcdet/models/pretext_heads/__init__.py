from .projection_point_head import ProjectionPointHead
from .seg_vox_head import SegVoxHead
from .reg_heads import RegHead
from .projection_sparse_vox_head import ProjectionSparseVoxHead



__all__ = {
    'ProjectionPointHead': ProjectionPointHead,
    'ProjectionSparseVoxHead': ProjectionSparseVoxHead,
    'SegVoxHead': SegVoxHead,
    'RegHead': RegHead
    }
