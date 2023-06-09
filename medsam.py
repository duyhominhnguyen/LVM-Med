from segmentation_2d.MedSAM import medsam_2d
from segmentation_3d.MedSAM import medsam_3d
from utils.func import (
    parse_config,
    load_config
)

if __name__=="__main__":
    yml_args = parse_config()
    cfg = load_config(yml_args.config)
    
    assert cfg.base.is_2D + cfg.base.is_3D == 1
    if cfg.base.is_2D:
        medsam_2d(yml_args, cfg)
    if cfg.base.is_3D:
        medsam_3d(yml_args, cfg)