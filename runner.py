from tqdm import tqdm

from utils.args_parser import parse_args
from utils.bench.dataloader import (
    create_data_loader,
    create_ids,
    create_reverse_ids,
    load_data,
    parse_nodes_to_ids,
    split_leave_one_out,
)
from utils.config import load_config
from utils.logger import setup_logger

args = parse_args()
cfg = load_config(args.config)
log = setup_logger()


class Runner:
    def __init__(self) -> None:
        log.info("Loading data.")
        self.users, self.song_dict, self.nodes = load_data()

        log.info("Creating mappings for user and item ids.")
        self.user2id, self.item2id = create_ids(self.users, self.song_dict)

        log.info("Creating reverse id mappings.")
        self.id2user, self.id2item = create_reverse_ids(
            self.user2id, self.item2id)

        log.info("Parsing nodes.")
        self.parsed_nodes = parse_nodes_to_ids(
            self.nodes, self.user2id, self.item2id)

        log.info("Runner instance created.")
        train, val, test, user_history = split_leave_one_out(self.parsed_nodes)
        self.train_loader = create_data_loader(
            train, self.item2id, user_history, True)
        self.val_loader = create_data_loader(
            val, self.item2id, user_history, False)
        self.test_loader = create_data_loader(
            test, self.item2id, user_history, False)

        self.loss = ...
        self.optimizer = ...

    def train(self) -> None:
        for user, pos_item, neg_item in tqdm(self.train_loader):
            ...

    def test(self) -> None:
        pass


if __name__ == "__main__":
    log.info("***** TuneGraph training bench *****")

    args = parse_args()

    runner = Runner()

    if args.mode == "train":
        runner.train()
    else:
        runner.test()
