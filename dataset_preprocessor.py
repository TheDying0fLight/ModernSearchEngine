from transformers import BertTokenizer, BertModel
import torch
import functorch
from datasets import load_dataset
import sys
import time
import json
from abc import ABC, abstractmethod
import os
import re

class DatasetPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, dataset, prefix, result_path):
        pass

# dataset preprocessor for msmarco and bert (expects basic msmarco dataset structure)
class MSMBertDatasetPreprocessor(DatasetPreprocessor):
    def __init__(self, bert_model_name=None):
        if bert_model_name is None:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        else:
            self.bert_model = BertModel.from_pretrained(bert_model_name)
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    def preprocess(self, dataset, prefix="", result_path="datasets/preprocessed/"):
        self.prefix = prefix # assign object var here, so other methods can access it without blowing up the passed parameters. This is not clean. Fix later.
        self.result_path = result_path # assign object var here, so other methods can access it without blowing up the passed parameters. This is not clean. Fix later.
        result_dir_contents = os.listdir(result_path)
        
        if not any(map(lambda x: f"{self.prefix}passages.json"==x, result_dir_contents)): # passages json is not already present. compute from scratch.
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

        passage_embedding_files = [
            passage_embedding_path
            for passage_embedding_path in result_dir_contents
            if f"{self.prefix}passage_embeddings" in passage_embedding_path
        ]
        if len(passage_embedding_files) > 0:
            max_processed_idx = max(map(lambda x: int(x.split("_")[-1].split(".")[0].split("-")[-1]), passage_embedding_files)) # horrible I know
        else:
            max_processed_idx = 0
        # encode passages
        with torch.no_grad(): #this preprocessing is static and not used for fine tuning, so no_grad saves resources
            passage_embeddings = self.bert_encode(passages[max_processed_idx:], checkpoint_save=10, last_saved_embedding=max_processed_idx)
        torch.save(passage_embeddings, f"{self.result_path}{self.prefix}passage_embeddings.pt")

        # get query encodings and save them
        queries = dataset["query"]
        with torch.no_grad(): #this preprocessing is static and not used for fine tuning, so no_grad saves resources
            query_embeddings = self.bert_encode(queries)
        torch.save(query_embeddings, f"{self.result_path}{self.prefix}query_embeddings.pt")

        # Make a dict mapping query ids to tensor indices
        query_ids = dataset["query_id"]
        query_id_map = {}
        for idx, query_id in enumerate(query_ids):
            if query_id in query_id_map:
                raise Exception("Duplicated Query ID. Core assumption violated")
            query_id_map[query_id] = idx
        with open(f"{self.result_path}{self.prefix}query_id_map.json", "w") as f:
            json.dump(query_id_map, f)

    def extract_passages_and_relevance(self, dataset): # slow for loop implementation. But not worth the effort to optimize rn. fix if nessecary
        passages = []
        relevance_assignments = []
        for idx in range(0, len(dataset["passages"])):
            relevance_labels = dataset["passages"][idx]["is_selected"]
            passage_list = dataset["passages"][idx]["passage_text"]

            passages.extend(dataset["passages"][idx]["passage_text"])
            relevance_assignments.extend([[relevance_labels[i], dataset["query_id"][i], passage_list[i]] for i in range(0, len(relevance_labels))])

        passages = list(set(passages))
        return passages, relevance_assignments

    def bert_encode(self, text, batch_size = 100, verbose = True, checkpoint_save=10, last_saved_embedding=0):
        batch_start = 0
        # inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True) # for the queries this works well, but the passages are waaaay too long and it takes forever
        all_embeddings = []
        
        # monitoring vars
        batch_times = []
        mem_used = 0
        completed_batches = 0
        total_batches = int(len(text)/batch_size)
        if len(text)%batch_size != 0:
            total_batches += 1

        while batch_start < len(text):
            if verbose:
                print("------------------------------------")

            start_time = time.time() # timing for eta calculation
            # quick and dirty batching so this doesnt crash my pc
            batch_end = min(batch_start + batch_size, len(text))
            batch = self.tokenizer(text[batch_start:batch_end], return_tensors="pt", padding=True, truncation=True)
            if verbose:
                print(f"#tokens this batch: {batch["input_ids"].shape[0]*batch["input_ids"].shape[1]}")
                print(f"#unmasked-tokens this batch: {torch.sum(batch["attention_mask"])}")
            #batch = inputs[batch_start:batch_end]

            outputs = self.bert_model(**batch)
            cls_token = outputs.pooler_output

            all_embeddings.append(cls_token) 

            batch_start += batch_size
            completed_batches += 1
            
            # monitoring calculations
            end_time = time.time()
            batch_times.append(end_time-start_time)
            mem_used += cls_token.nelement()*cls_token.element_size()/10**9 # memory usage of new tensor in GB
            remaining_batches = total_batches-completed_batches

            # monitoring print
            if verbose:
                print(f"Remaining time: \t \t {round(remaining_batches*(sum(batch_times)/len(batch_times))/60, 3)} min")
                print(f"Completion percentage: \t \t {round((100*len(all_embeddings)*all_embeddings[0].shape[0]/len(text)), 3)}%") # currently broken
                print(f"Memory used: \t \t \t {round(mem_used, 4)}GB")
                print(f"Projected total memory use: \t {round(total_batches*mem_used/completed_batches, 4)}GB")

            if checkpoint_save > 0 and completed_batches%checkpoint_save == 0: # save every checkpoint_save batches
                all_embeddings = torch.cat(all_embeddings, dim=0)
                torch.save(all_embeddings, f"{self.result_path}{self.prefix}passage_embeddings_{last_saved_embedding}-{last_saved_embedding+batch_size*batch_times}.pt")
                all_embeddings = [] #reset embedding buffer to free up space
                last_saved_embedding += batch_size*checkpoint_save

        return all_embeddings 

if __name__ == "__main__":
    ds = load_dataset("microsoft/ms_marco", "v1.1")
    ds.with_format("torch")
    preprocessor = MSMBertDatasetPreprocessor()
    preprocessor.preprocess(ds["train"][:], prefix="[train]", result_path="datasets/preprocessed/")