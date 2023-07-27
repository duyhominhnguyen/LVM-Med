from segmentation_2d.MedSAM_2d import medsam_2d
from segmentation_2d.LVMMed_SAM_2d import lvm_medsam_2d

from segmentation_3d.MedSAM_3d import medsam_3d
from segmentation_3d.LVMMed_SAM_3d import lvm_medsam_3d

from utils.func import (
    parse_config,
    load_config
)

if __name__=="__main__":
    yml_args = parse_config()
    cfg = load_config(yml_args.config)
    
    assert cfg.base.is_2D + cfg.base.is_3D == 1

    if yml_args.lvm_med_encoder_path != '':
        if cfg.base.is_2D:
            lvm_medsam_2d(yml_args, cfg)
        else:
            lvm_medsam_3d(yml_args, cfg)
    else:
        if cfg.base.is_2D:
            medsam_2d(yml_args, cfg)
        if cfg.base.is_3D:
            medsam_3d(yml_args, cfg)