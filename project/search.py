import torch
import os
import json
from tqdm import tqdm
import numpy as np
from project import SiglipStyleModel, ColSentenceModel, DOCS_FILE

class SearchEngine():
    def __init__(self, data_folder="../data", embedding_file:str="embeddings.pkl"):
        self.embedding_dict = self._load_embeddings(path=os.path.join(data_folder, embedding_file))
        self.docs = self._load_docs(path=os.path.join(data_folder, DOCS_FILE))
        # model = ColSentenceModel().load(r"project\retriever\model_uploads\bmini_ColSent_b128_marco_v1.safetensors")
        self.model: SiglipStyleModel | ColSentenceModel = SiglipStyleModel().load(r"project/retriever/model_uploads/bert-mini_b32_marco_v1.safetensors")

    def _load_embeddings(self, path: str = "../data/embeddings.pkl") -> dict[torch.Tensor, str]:
        return torch.load(path)

    def _load_docs(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"DOC file not found at {path}")

        docs = {}
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, "Line"):
                doc = json.loads(line.strip())
                docs[doc["url"]] = doc
        return docs

    def retrieve(self, query: str):
        similarities = []
        query_embedding = self.model.embed(query)
        for embedding, _ in tqdm(list(self.embedding_dict.items()), "Similarities"):
            similarity = self.model.resolve(query_embedding, embedding.cuda()).squeeze()
            similarities.append(similarity.detach().cpu())
        vals = np.array(list(zip(self.embedding_dict.values(), similarities)))
        return vals[np.argsort(similarities)[::-1]]

    def search(self, query, max_res=100):
        res = self.retrieve(query)[:max_res]
        return [self.docs[r[0]] for r in res]