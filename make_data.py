from argparse import ArgumentParser

from src.data.convert_rook import make_policy_bc_data, make_policy_sv_data, make_policy_av_data

parser = ArgumentParser("Convert ROOK data to RookWorld policy training data")
parser.add_argument("-i", "--input", help="Input text file with ROOK data format")
parser.add_argument("-p", "--predictor", choices=["bc", "av", "sv"], help="Convert to bc|av|sv policy data")
parser.add_argument("-kd", "--kd", action="store_true", help="Add probabilities for all moves")
parser.add_argument("-m", "--mtl", action="store_true", help="Add MTL task token")
parser.add_argument("-c", "--cot", action="store_true", help="Add COT data")
#parser.add_argument("-o", "--output", help="Output file")
args = parser.parse_args()

if __name__ == "__main__":
    with open(args.input, "r") as f:
        if args.predictor == "bc":
            dataset = make_policy_bc_data(f, mtl=args.mtl, cot=args.cot, probas=args.probas, debug=True)
        elif args.predictor == "sv":
            dataset = make_policy_sv_data(f)
        elif args.predictor == "av":
            dataset = make_policy_av_data(f)
        else:
            raise ValueError("Unknown predictor type")
