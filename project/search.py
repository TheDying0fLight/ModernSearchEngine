import torch
import os
import json
import nltk
import re
import logging
from nltk.corpus import stopwords
from tqdm import tqdm
import numpy as np
from sklearn.base import ClusterMixin
from project import SiglipStyleModel, ColSentenceModel, DocumentCollection, Document
from bs4 import BeautifulSoup
from typing import Dict, Set

class SearchEngine():
    def __init__(self, data_folder="data", embedding_file:str="embeddings.pkl", cluster_embedding_file:str="clustering_embeddings.pkl"):
        self.embedding_dict: Dict[torch.Tensor, str] = self._load_embeddings(path=os.path.join(data_folder, embedding_file))
        self.cluster_embedding_dict: Dict[str, torch.Tensor] = self._load_embeddings(path=os.path.join(data_folder, cluster_embedding_file))
        self.docs: DocumentCollection = self._load_docs(path=data_folder)
        self.stop_words: Set[str] = self._load_stop_words()
        self.model = ColSentenceModel().load(r"project\retriever\model_uploads\bmini_ColSent_b128_marco_v1.safetensors")
        #self.model: SiglipStyleModel | ColSentenceModel = SiglipStyleModel().load(r"project/retriever/model_uploads/bert-mini_b32_marco_v1.safetensors")

    def _load_embeddings(self, path: str) -> dict[torch.Tensor, str]:
        return torch.load(path)

    def _load_docs(self, path: str) -> DocumentCollection:
        if not os.path.exists(path):
            raise FileNotFoundError(f"DOC file not found at {path}")

        docs = DocumentCollection()
        docs.load_from_file(dir_path=path, load_html=False)
        return docs

    def _load_stop_words(self):
        nltk.download('stopwords')
        return set(stopwords.words('english'))


    def get_sentence_wise_similarities(self, query_embedding, relevant):
        similarities = {}
        for url, embedding in relevant:
            sentence_similarity = self.model.sentence_sim(torch.tensor(embedding).cuda(), query_embedding).squeeze()
            similarities[url] = sentence_similarity.detach().cpu().tolist()
        return similarities


    def retrieve(self, query: str):
        similarities = []
        query_embedding = self.model.embed(query)[0]
        for embedding, _ in tqdm(list(self.embedding_dict.items()), "Similarities"):
            similarity = self.model.resolve(query_embedding, embedding.cuda()).squeeze()
            similarities.append(similarity.detach().cpu())
        urls = np.array(list(self.embedding_dict.values()))
        embeddings = list(self.embedding_dict.keys())
        similarities = np.array(similarities)
        sorted_sim_index = np.argsort(-similarities)
        return urls[sorted_sim_index], [embeddings[i] for i in sorted_sim_index], similarities[sorted_sim_index], query_embedding


    def search(self, query: str, max_res=100):
        """Filter and preprocess query for better results and search with the model"""
        orig_query = query
        if not re.search(r't[^h\-\s]{1,6}bingen', query.lower()): # add tübingen if it is not part of the query
            query += ' tuebingen'
        filtered_query = ' '.join([word for word in query.split() if word not in self.stop_words] ) # filter query for stop words
        (urls, embeddings, similarities, query_embedding) = self.retrieve(filtered_query) # retrieve results from model
        sentence_wise_similarities = self.get_sentence_wise_similarities(query_embedding, zip(urls[:max_res], embeddings[:max_res])) # get sentence wise similarity for best results
        logging.info(f'Searched for "{filtered_query}" (original: "{orig_query}")')
        return urls[:max_res], [self.docs.documents[url] for url in urls[:max_res]], similarities[:max_res], sentence_wise_similarities


    def cluster_topics(self, docs: list[Document], embeddings: np.ndarray, scores: np.ndarray, clustering_alg: ClusterMixin):
        """Clusters the results into topics. Topics are sorted by the max score. In the topics documents are sorted by the score.

        Args:
            docs (list[Document]): list of documents. Already sorted by scores!
            embeddings (np.ndarray): embeddings of the docs
            scores (np.ndarray): scores of the docs
            clustering_alg (ClusterMixin): clustering algorithm
        """
        labels = clustering_alg.fit_predict(embeddings)
        num_topics = len(set(labels))
        topics = [[] for _ in range(num_topics)]
        topic_scores = [[] for _ in range(num_topics)]
        for doc, score, label in zip(docs, scores, labels):
            topics[label].append(doc)
            topic_scores[label].append(score)
        topic_max_scores = np.array([max(s) for s in topic_scores])
        sorted_topic_scores = np.argsort(-topic_max_scores)
        return [topics[i] for i in sorted_topic_scores]

    def search_and_cluster(self, query, clustering_alg: ClusterMixin, max_res=100):
        urls, docs, scores, sentence_wise_similarities = self.search(query, max_res)
        doc_embeddings = [self.cluster_embedding_dict[url] for url in urls]
        docs_by_topics = self.cluster_topics(docs, doc_embeddings, scores, clustering_alg)
        return docs_by_topics, sentence_wise_similarities
