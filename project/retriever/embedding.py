import torch
from torch.utils.data import Dataset
import json

# used to make large embeddings feasible by exporting some to disk
class Embedding(Dataset):
    def __init__(self, data_path):
        self.batch_file_paths = []
        self.batch_idxs = []
        self.data_path = data_path
        self.last_loaded_batch = None
        self.last_loaded_batch_edges = (-1, -1)

    def add(self, batch_tensor): # add new embedding batch at the END of the overall embedding list
        batch_tensor_path = self.data_path+f"batch_{len(self.batch_file_paths)}.pt"
        torch.save(batch_tensor, batch_tensor_path)
        if len(self.batch_idxs) > 0:
            self.batch_idxs.append(self.batch_idxs[-1]+len(batch_tensor))
        else:
            self.batch_idxs.append(len(batch_tensor))

    def __len__(self):
        if len(self.batch_idxs) > 0:
            return self.batch_idxs[-1]
        else:
            return 0
    
    def __getitem__(self, idx):
        if idx <= self.last_loaded_batch_edges[0] or idx > self.last_loaded_batch_edges[1]:
            file_idx = 0
            file_start_offset = 0
            while idx > self.batch_idxs[file_idx]:
                file_start_offset = self.batch_idxs[file_idx]
                file_idx += 1
                if file_idx >= len(self.batch_idxs):
                    raise IndexError()
            self.last_loaded_batch = torch.load(self.data_path+f"batch_{file_idx}.pt")
            self.last_loaded_batch_edges = (file_start_offset, self.batch_idxs[file_idx])
            
        batch_tensor = self.last_loaded_batch
        local_idx = idx - self.last_loaded_batch_edges[0]
        return batch_tensor[local_idx]
    
    def save(self):
        save_dict = {
            "file_paths": self.batch_file_paths,
            "batch_idxs": self.batch_idxs,
        }

        with open(self.data_path+"embedding.json", "w") as f:
            json.dump(save_dict, f)
    
    def load(self): # doesnt have safeguards to prevent loading files that dont exist. Expects user to only call this if a valid embedding file is present
        with open(self.data_path+"embedding.json", "r") as f:
            save_dict = json.load(f)
        self.batch_file_paths = save_dict["file_paths"]
        self.batch_idxs = save_dict["batch_idxs"]
            
            