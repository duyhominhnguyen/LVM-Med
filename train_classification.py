from classification_R50.train_R50_classification_FGADR import train_FGADR_R50

from utils.func import (
    parse_config,
    load_config
)
if __name__=="__main__":
    yml_args = parse_config()
    cfg = load_config(yml_args.config)
 
    assert cfg.base.is_R50 + cfg.base.is_SAMVIT == 1
    if cfg.base.dataset_name == "fgadr":
        if cfg.base.is_R50:
            train_FGADR_R50(cfg)
    else:
        print("Wrong")