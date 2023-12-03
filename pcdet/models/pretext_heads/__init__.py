from .seg_head import SegHead
from .seg_vox_head import SegVoxHead
from .depth_head import DepthHead
from .reg_heads import RegHead
# from .seg_sparse_vox_head import SegSparseVoxHead



__all__ = {
    'SegHead': SegHead,
    'SegVoxHead': SegVoxHead,
    # 'SegSparseVoxHead': SegSparseVoxHead,
    'DepthHead': DepthHead,
    'RegHead': RegHead
    }
