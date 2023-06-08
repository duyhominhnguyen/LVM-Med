import yaml
from munch import munchify
import argparse 

def load_config(path):
    with open(path, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    return munchify(cfg)

def parse_config():
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument(
        '-config',
        type=str,
        default='./configs/default.yaml',
        help='Path to the config file.'
    )
    parser.add_argument(
        '-overwrite',
        action='store_true',
        default=False,
        help='Overwrite file in the save path.'
    )
    parser.add_argument(
        '-print_config',
        action='store_true',
        default=False,
        help='Print details of configs.'
    )
    parser.add_argument(
        '-test', 
        '--use_test_mode', 
        type=int, 
        default=0, 
        help='Flag for inference only (no train mode)')
    args = parser.parse_args()
    return args