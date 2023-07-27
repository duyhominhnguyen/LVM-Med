from segmentation_2d.zero_shot_SAM_2d import zero_shot_sam_2d
from segmentation_2d.zero_shot_LVMMed_SAM_2d import zero_shot_lvmmed_sam_2d

from utils.func import (
    parse_config,
    load_config
)

if __name__=="__main__":
    yml_args = parse_config()
    cfg = load_config(yml_args.config)

    assert cfg.base.is_2D + cfg.base.is_3D == 1
    
    if cfg.base.is_3D:
        assert NotImplementedError(f"[Error] We have not yet implemented this function for 3D dataset. You could try implement this similarly based on our 3D implementation for MedSAM.")
        
    if yml_args.lvm_med_encoder_path != '':
        zero_shot_lvmmed_sam_2d(yml_args, cfg)
    else:
        zero_shot_sam_2d(yml_args, cfg)