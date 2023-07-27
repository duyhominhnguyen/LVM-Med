from classification_R50.train_R50_classification import train_R50

from utils.func import (
    parse_config,
    load_config
)
if __name__=="__main__":
    yml_args = parse_config()
    cfg = load_config(yml_args.config)
 
    assert cfg.base.is_R50 + cfg.base.is_SAMVIT == 1
    
    if cfg.base.is_R50:
        train_R50(yml_args, cfg)
    else:
        print("Wrong")