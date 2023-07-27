import argparse
from datasets_split import Kvasir_split, BUID_split, FGADR_split, MMWHS_MR_Heart_split, MMWHS_CT_Heart_split 

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--dataset_name', '-ds', metavar='DS', type=str, default="", help='Name of dataset')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    if args.dataset_name == "Kvasir":
        Kvasir_split.Kvasir_split()
    elif args.dataset_name == "BUID":
        BUID_split.BUID_split()
    elif args.dataset_name == "FGADR":
        FGADR_split.FGADR_split()
    elif args.dataset_name == "MMWHS_MR_Heart":
        MMWHS_MR_Heart_split.MMWHS_MR_Heart_split()
    elif args.dataset_name == "MMWHS_CT_Heart":
        MMWHS_CT_Heart_split.MMWHS_CT_Heart_split()
    else:
        print("Let's input dataset name")