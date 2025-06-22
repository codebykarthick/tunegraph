from utils.args_parser import parse_args
from utils.config import load_config
from utils.logger import setup_logger

args = parse_args()
cfg = load_config(args.config)
logger = setup_logger()


class Runner:
    def __init__(self) -> None:
        pass

    def train(self) -> None:
        pass

    def test(self) -> None:
        pass


if __name__ == "__main__":
    logger.info("***** TuneGraph training bench *****")
    runner = Runner()

    if args.mode == "train":
        runner.train()
    else:
        runner.test()
