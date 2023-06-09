from segmentation_2d.MedSAM import medsam_2d
from utils.func import (
    parse_config,
    load_config
)

if __name__=="__main__":
    yml_args = parse_config()
    cfg = load_config(yml_args.config)
    if yml_args.use_2D:
        medsam_2d(yml_args, cfg)
    
