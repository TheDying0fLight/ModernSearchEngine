import numpy as np
from datasets import load_dataset

def collate_fn(batch):
    def get_relevant(passages):
        passage = np.array(passages["passage_text"])[np.array(passages["is_selected"], bool)]
        return passage[0] if len(passage) > 0 else ""
    res = {
        "query": [ex["query"] for ex in batch],
        # "answer": [ex["answers"][0] if ex["answers"] else "" for ex in batch],
        "answer": [get_relevant(ex["passages"]) for ex in batch],
    }
    return res

def get_train_and_test_data(data_paths: list[tuple[str]]):
    test_data = {}
    for path, version in data_paths:
        name = path + version if version else ""
        test_data[name] = load_dataset(path, version, split="test").take(5000)
    train_data = load_dataset(data_paths[0][0], data_paths[0][1], split="train")
    return train_data, test_data