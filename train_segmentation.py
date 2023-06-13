from segmentation_2d.train_R50_seg_adam_optimizer_2d import train_2d_R50
from segmentation_2d.train_SAM_VIT_seg_adam_optimizer_2d import train_2d_SAMVIT
from segmentation_3d.train_R50_seg_adam_optimizer_3d import train_3d_R50
from segmentation_3d.train_SAM_VIT_seg_adam_optimizer_3d import train_3d_SAMVIT
from utils.func import (
    parse_config,
    load_config
)
if __name__=="__main__":
    yml_args = parse_config()
    cfg = load_config(yml_args.config)
 
    assert cfg.base.is_2D + cfg.base.is_3D == 1
    if cfg.base.is_2D:
        if cfg.base.is_R50:
            train_2d_R50(cfg)
        if cfg.base.is_SAMVIT:
            train_2d_SAMVIT(cfg)
    if cfg.base.is_3D:
        if cfg.base.is_R50:
            train_3d_R50(cfg)
        if cfg.base.is_SAMVIT:
            train_3d_SAMVIT(cfg)