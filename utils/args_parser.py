import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Add anymore arguments needed here
    parser.add_argument("--config", "-c", type=str,
                        default=None, help="The path to config file to use for program.")
    parser.add_argument("--mode", "-m", type=str, choices=["train", "test"], required=False,
                        help="Mode to be used for the training bench.")

    args = parser.parse_args()

    return args
