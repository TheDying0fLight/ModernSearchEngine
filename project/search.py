import torch
import os
import json
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import numpy as np
from project import SiglipStyleModel, ColSentenceModel, DocumentCollection
from sklearn.cluster import AffinityPropagation
from bs4 import BeautifulSoup
from typing import Dict, Set

class SearchEngine():
    def __init__(self, data_folder="data", embedding_file:str="embeddings.pkl"):
        self.embedding_dict: Dict[torch.Tensor, str] = self._load_embeddings(path=os.path.join(data_folder, embedding_file))
        self.docs: DocumentCollection = self._load_docs(path=data_folder)
        # self._load_snippets(path=os.path.join(data_folder, HTML_FILE))
        self.stop_words: Set[str] = self._load_stop_words()
        # model = ColSentenceModel().load(r"project\retriever\model_uploads\bmini_ColSent_b128_marco_v1.safetensors")
        self.model: SiglipStyleModel | ColSentenceModel = SiglipStyleModel().load(r"project/retriever/model_uploads/bert-mini_b32_marco_v1.safetensors")

    def _load_embeddings(self, path: str) -> dict[torch.Tensor, str]:
        return torch.load(path)

    def _load_docs(self, path: str) -> DocumentCollection:
        if not os.path.exists(path):
            raise FileNotFoundError(f"DOC file not found at {path}")

        docs = DocumentCollection()
        docs.load_from_file(dir_path=path, load_html=False)
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


    def _load_stop_words(self) -> Set[str]:
        nltk.download('stopwords')
        return set(stopwords.words('english'))

    def get_sentence_wise_similarities(self, query_embedding, relevant_urls):
        similarities = {}
        for url in relevant_urls:
            doc_embedding = self.embedding_dict[url]
            sentence_similarity = self.model.sentence_sim(query_embedding, doc_embedding.cuda()).squeeze()
            similarities[url] = sentence_similarity.detach().cpu().tolist()
        return similarities

    def retrieve(self, query: str, max_res=100):
        similarities = []
        query_embedding = self.model.embed(query)
        for embedding, _ in tqdm(list(self.embedding_dict.items()), "Similarities"):
            similarity = self.model.resolve(query_embedding, embedding.cuda()).squeeze()
            similarities.append(similarity.detach().cpu())
        urls = np.array(list(self.embedding_dict.values()))
        embeddings = np.array(list(self.embedding_dict.keys()))
        similarities = np.array(similarities)
        sorted_sim_index = np.argsort(-similarities)
        relevant_urls = urls[sorted_sim_index[:max_res]]
        sentence_wise_similarities = self.get_sentence_wise_similarities(query_embedding, relevant_urls)
        return urls[sorted_sim_index], embeddings[sorted_sim_index], similarities[sorted_sim_index], sentence_wise_similarities

    def search(self, query: str, max_res=100):
        filtered = [word for word in query.split() if word not in self.stop_words]
        filtered_query = ' '.join(filtered)
        urls, embeddings, similarities, sentence_wise_similarities = self.retrieve(filtered_query, max_res=max_res)
        return [self.docs.documents[url] for url in urls[:max_res]], embeddings[:max_res], similarities[:max_res], sentence_wise_similarities

    def search_and_cluster(self, query, max_res=100):
        docs, embeddings, scores, sentence_wise_similarities = self.search(query, max_res)
        labels = AffinityPropagation().fit_predict(embeddings)
        num_topics = len(set(labels))
        topics = [[] for _ in range(num_topics)]
        topic_scores = [[] for _ in range(num_topics)]
        for doc, score, label in zip(docs, scores, labels):
            topics[label].append(doc)
            topic_scores[label].append(score)
        topic_max_scores = np.array([max(s) for s in topic_scores])
        sorted_topic_scores = np.argsort(-topic_max_scores)
        return [topics[i] for i in sorted_topic_scores], sentence_wise_similarities