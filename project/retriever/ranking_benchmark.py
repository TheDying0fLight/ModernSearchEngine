from datasets import load_dataset
import numpy as np
from model import BM25
import os
import torch
import json


class ranking_benchmark:
    def __init__(self, dataset_name, dir_name, prefix="", result_path="datasets/preprocessed/"):
        dataset = load_dataset(dataset_name, dir_name)
        self.bm25 = BM25()
        self.queries = {}

        for query_idx in range(len(dataset["test"]["queries"])):
            self.queries[dataset["test"]["queries"][query_idx]] = dataset["test"]["query_id"][query_idx]

        self.documents, self.relevance_map = self.preprocess(dataset["test"], prefix, result_path)
        self.bm25.preprocess(self.documents)  # computes bow components ahead of time

        self.per_query_rankings = {}

    def preprocess(self, dataset, prefix="", result_path="datasets/preprocessed/"):
        # assign object var here, so other methods can access it without blowing up the passed parameters. This is not clean. Fix later.
        self.prefix = prefix
        # assign object var here, so other methods can access it without blowing up the passed parameters. This is not clean. Fix later.
        self.result_path = result_path
        result_dir_contents = os.listdir(result_path)

        # passages json is not already present. compute from scratch.
        if not any(map(lambda x: f"{self.prefix}passages.json" == x, result_dir_contents)):
            passages, relevance_assignments = self.extract_passages_and_relevance(dataset)

            # save relevance assignments
            passage_map = {}
            for idx, passage in enumerate(passages):
                passage_map[passage] = idx
            relevance_assignments = list(map(lambda x: [x[0], x[1], passage_map[x[2]]], relevance_assignments))
            relevance_tensor = torch.tensor(relevance_assignments)
            torch.save(relevance_tensor, f"{self.result_path}{self.prefix}relevance_assignments.pt")

            # save list of passages (NOTE: we treat passage ids implicitly by referencing the index in this list. DO NOT SHUFFLE.)
            with open(f"{self.result_path}{self.prefix}passages.json", "w") as f:
                json.dump(passages, f)
        else:
            with open(f"{self.result_path}{self.prefix}passages.json", "r") as f:
                passages = json.load(f)

        return passages, relevance_assignments

    # slow for loop implementation. But not worth the effort to optimize rn. fix if nessecary
    def extract_passages_and_relevance(self, dataset):
        passages = []
        relevance_assignments = []
        for idx in range(0, len(dataset["passages"])):
            relevance_labels = dataset["passages"][idx]["is_selected"]
            passage_list = dataset["passages"][idx]["passage_text"]

            passages.extend(dataset["passages"][idx]["passage_text"])
            relevance_assignments.extend([[relevance_labels[i], dataset["query_id"][i], passage_list[i]]
                                         for i in range(0, len(relevance_labels))])

        passages = list(set(passages))
        return passages, relevance_assignments

    def get_per_query_rankings(self):
        for query in self.queries.keys():
            relevance_assignments = np.vstack((self.bm25.calculate_rels(query), np.arange(len(self.documents))))
