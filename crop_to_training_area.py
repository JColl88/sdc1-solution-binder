import argparse

from ska.sdc1.utils.image_utils import crop_to_training_area

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Path to input image", type=str)
    parser.add_argument("-o", help="Path to output image", type=str)
    parser.add_argument(
        "-f", help="Image frequency band (560||1400||9200, MHz)", default=1400, type=int
    )
    parser.add_argument(
        "-p", help="Padding factor to include edges", default=1.0, type=float
    )
    args = parser.parse_args()

    crop_to_training_area(args.i, args.o, args.f, args.p)
