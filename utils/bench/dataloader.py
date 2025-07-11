import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from torch.utils.data import DataLoader
from tqdm import tqdm

from entities.data import TrackMetadata
from utils.bench.dataset import BPRDataset
from utils.config import load_config
from utils.logger import setup_logger

cfg = load_config()
log = setup_logger()


def load_data() -> Tuple[List[int], Dict[str, TrackMetadata], List[Tuple[int, str]]]:
    """
    Load Spotify MPD data.

    Returns:
        Tuple[List[int], Dict[str, TrackMetadata], List[Tuple[int, str]]] representing
            (user_ids, song_dict, edges).
    """
    data_path = os.path.join(
        os.getcwd(), "data", "spotify_million_playlist_dataset", "data")

    # Maps song id to artist, album and song if needed
    song_dict: Dict[str, TrackMetadata] = {}

    # The unique PIDs treated as different users (not ideal)
    users: List[int] = []

    # The connection between PIDs and song IDs
    nodes: List[Tuple[int, str]] = []

    file_list = sorted(os.listdir(data_path))
    slice_limit = 1 if cfg["data"]["limited"] else None
    files = [os.path.join(data_path, file) for file in file_list[:slice_limit]]

    for file in tqdm(files, desc="Files processed"):
        with open(file, "r") as f:
            try:
                doc = json.load(f)
                for playlist in tqdm(doc["playlists"], desc="Current File"):
                    if not playlist["tracks"]:
                        continue
                    pid: int = playlist["pid"]
                    users.append(pid)
                    for track in playlist["tracks"]:
                        track_id = track["track_uri"]
                        artist_name = track["artist_name"]
                        track_name = track["track_name"]
                        album_name = track["album_name"]

                        if track_id not in song_dict:
                            # Add new song to songs list
                            song_dict[track_id] = TrackMetadata(
                                artist_name, track_name, album_name)

                        nodes.append((pid, track_id))
            except json.JSONDecodeError as e:
                log.fatal(f"Failed to parse {file}: {e}")

    return users, song_dict, nodes


def create_ids(users: List[int], song_dict: Dict[str, TrackMetadata]) -> Tuple[Dict[int, int], Dict[str, int]]:
    """Map the string IDs to integer IDs for easier training.

    Args:
        users (List[str]): The list of unique users in the dataset.
        song_dict (Dict[str, TrackMetadata]): The unique song dictionary generated

    Returns:
        Tuple[Dict[str, int], Dict[str, int]]: The to id mapping dictionary for both user and item.
    """
    user2id = {pid: idx for idx, pid in enumerate(sorted(set(users)))}
    item2id = {uri: idx for idx, uri in enumerate(song_dict.keys())}

    return user2id, item2id


def create_reverse_ids(user2id: Dict[int, int], item2id: Dict[str, int]) -> Tuple[Dict[int, int], Dict[int, str]]:
    """Given the direct to mapping, create reverse mapping for inference time.

    Args:
        user2id (Dict[str, int]): The user to id mapping.
        item2id (Dict[str, int]): The item to id mapping.

    Returns:
        Tuple[Dict[int, str], Dict[int, str]]: The reverse id mapping dictionary for both user and item.
    """
    id2user = {v: k for k, v in user2id.items()}
    id2item = {v: k for k, v in item2id.items()}

    return id2user, id2item


def parse_nodes_to_ids(nodes: List[Tuple[int, str]], user2id: Dict[int, int], item2id: Dict[str, int]) -> List[Tuple[int, int]]:
    """Process the nodes to mapped ids.

    Args:
        nodes (List[Tuple[str, str]]): The user item mapped ids
        user2id (Dict[str, int]): The dictionary mapping user to ids
        item2id (Dict[str, int]): The dictionary mapping item to ids.

    Returns:
        List[Tuple[int, int]]: The parsed nodes of the graph mapping user id to item id.
    """
    parsed_nodes = [(user2id[user], item2id[item]) for (user, item) in nodes]

    return parsed_nodes


def split_leave_one_out(edges: List[Tuple[int, int]],
                        seed: int = 42) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]],
                                                 List[Tuple[int, int]], Dict[int, Set[int]]]:
    """Creates the train, validation test sets and user history.

    Args:
        edges (List[Tuple[int, int]]): The parsed user item edges
        seed (int, optional): The seed for split. Defaults to 42.

    Returns:
        Tuple[List[Tuple[int, int]], List[Tuple[int, int]],
                List[Tuple[int, int]], Dict[int, List[int]]]: The train, validation, test and user history.
    """
    random.seed(seed)

    user_history = defaultdict(set)
    for user, item in edges:
        user_history[user].add(item)

    train, val, test = [], [], []

    for user, items in user_history.items():
        if len(items) < 3:
            # not enough data → all to train
            for item in items:
                train.append((user, item))
            continue

        items = list(items)
        random.shuffle(items)
        test_item = items.pop()
        val_item = items.pop()
        train_items = items

        train.extend((user, item) for item in train_items)
        val.append((user, val_item))
        test.append((user, test_item))

    return train, val, test, user_history


def create_data_loader(data: List[Tuple[int, int]],
                       item2id: Dict[str, int],
                       user_history: Dict[int, Set[int]],
                       is_train: bool) -> DataLoader[BPRDataset]:
    """Creates a dataloader from the given data, song dictionary and user history.

    Args:
        data (List[Tuple[int, int]]): The user, item node list
        song_dict (Dict[str, TrackMetadata]): The unique song dictionary
        user_history (Dict[int, List[int]]): The user and their item history. 
        is_train (bool): To switch between creating a large train loader or a small test/validation loader.

    Returns:
        DataLoader: The PyTorch dataloader instance.
    """
    dataset = BPRDataset(data, len(item2id), user_history)
    train_cfg = cfg["train"]

    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg["num_workers"] if is_train else 1

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=is_train)

    return dataloader
