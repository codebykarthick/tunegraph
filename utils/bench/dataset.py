import random
from typing import Dict, List, Set, Tuple

import torch
from torch.utils.data import Dataset


class BPRDataset(Dataset):
    def __init__(self, user_item_pairs: List[Tuple[int, int]], num_items, user_history: Dict[int, Set[int]]):
        self.user_item_pairs = user_item_pairs
        self.num_items = num_items
        self.user_history = user_history

    def __len__(self) -> int:
        return len(self.user_item_pairs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user, pos_item = self.user_item_pairs[idx]
        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in self.user_history[user]:
                break

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
        )
