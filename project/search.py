import torch
import os
import json
from tqdm import tqdm
import numpy as np
from project import SiglipStyleModel, ColSentenceModel, DOCS_FILE, HTML_FILE
from sklearn.cluster import AffinityPropagation
from bs4 import BeautifulSoup

class SearchEngine():
    def __init__(self, data_folder="data", embedding_file:str="embeddings.pkl"):
        self.embedding_dict = self._load_embeddings(path=os.path.join(data_folder, embedding_file))
        self.docs = self._load_docs(path=os.path.join(data_folder, DOCS_FILE))
        # self._load_snippets(path=os.path.join(data_folder, HTML_FILE))
        self.stop_words = self._load_stop_words(path=os.path.join(data_folder, "stopwords.txt"))
        # model = ColSentenceModel().load(r"project\retriever\model_uploads\bmini_ColSent_b128_marco_v1.safetensors")
        self.model: SiglipStyleModel | ColSentenceModel = SiglipStyleModel().load(r"project/retriever/model_uploads/bert-mini_b32_marco_v1.safetensors")

    def _load_embeddings(self, path: str) -> dict[torch.Tensor, str]:
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

    # def _load_snippets(self, path: str):
    #     with open(path, 'r', encoding='utf-8') as f:
    #         lines = f.readlines()
    #         for line in tqdm(lines, "Line"):
    #             doc: dict = json.loads(line.strip())
    #             url, html = list(doc.items())[0]
    #             if url in self.docs:
    #                 self.docs[url]['snippet'] = self._preprocess_html(html)


    # def _preprocess_html(self, html: str, seperator: str = '. ', max_char=100) -> str:
    #     soup = BeautifulSoup(html, 'html.parser')
    #     text = soup.get_text(separator=seperator, strip=True)
    #     return text.strip()[:max_char]


    def _load_stop_words(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f if line.strip()]
        return stopwords

    def retrieve(self, query: str):
        similarities = []
        query_embedding = self.model.embed(query)
        for embedding, _ in tqdm(list(self.embedding_dict.items()), "Similarities"):
            similarity = self.model.resolve(query_embedding, embedding.cuda()).squeeze()
            similarities.append(similarity.detach().cpu())
        urls = np.array(list(self.embedding_dict.values()))
        embeddings = np.array(list(self.embedding_dict.keys()))
        similarities = np.array(similarities)
        sorted_sim_index = np.argsort(-similarities)
        return urls[sorted_sim_index], embeddings[sorted_sim_index], similarities[sorted_sim_index]

    def search(self, query: str, max_res=100):
        filtered = [word for word in query.split() if word not in self.stop_words]
        filtered_query = ' '.join(filtered)
        urls, embeddings, similarities = self.retrieve(filtered_query)
        return [self.docs[url] for url in urls[:max_res]], embeddings[:max_res], similarities[:max_res]

    def search_and_cluster(self, query, max_res=100):
        docs, embeddings, scores = self.search(query, max_res)
        labels = AffinityPropagation().fit_predict(embeddings)
        num_topics = len(set(labels))
        topics = [[] for _ in range(num_topics)]
        topic_scores = [[] for _ in range(num_topics)]
        for doc, score, label in zip(docs, scores, labels):
            topics[label].append(doc)
            topic_scores[label].append(score)
        topic_max_scores = np.array([max(s) for s in topic_scores])
        sorted_topic_scores = np.argsort(-topic_max_scores)
        return [topics[i] for i in sorted_topic_scores]