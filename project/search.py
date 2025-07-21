import torch
import os
import json
import nltk
import re
import logging
from nltk.corpus import stopwords
from tqdm import tqdm
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.base import ClusterMixin
from bs4 import BeautifulSoup
from typing import Dict, Set

from project import SiglipStyleModel, ColSentenceModel, DocumentCollection, Document, BM25

device = "cuda" if torch.cuda.is_available() else "cpu"

class SearchEngine():
    def __init__(self, data_folder="data", embedding_file:str="embeddings.pkl", cluster_embedding_file:str="clustering_embeddings.pkl"):
        self.embedding_dict: Dict[str, torch.Tensor] = self._load_embeddings(path=os.path.join(data_folder, embedding_file))
        self.cluster_embedding_dict: Dict[str, torch.Tensor] = self._load_embeddings(path=os.path.join(data_folder, cluster_embedding_file))
        self.docs: DocumentCollection = self._load_docs(path=data_folder)
        self.stop_words: Set[str] = self._load_stop_words()
        self.model: ColSentenceModel = ColSentenceModel().load("project/retriever/model_uploads/bmini_ColSent_b128_marco_v1.safetensors")
        self.retriever_model = BM25().load("data/bm25_state.pkl")

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


    def get_sentence_wise_similarities(self, query_embedding, urls):
        similarities = {}
        for url in urls:
            embedding = self.embedding_dict[url]
            sentence_similarity = self.model.sentence_sim(embedding.to(device), query_embedding).squeeze()
            similarities[url] = sentence_similarity.detach().cpu().tolist()
        return similarities

    def retrieve(self, query: str):
        similarities = []
        similarities = self.retriever_model.resolve(query)
        urls = np.array(list(similarities.keys()))
        similarities = np.array(list(similarities.items()))
        sorted_sim_index = np.argsort(-similarities)
        return urls[sorted_sim_index], similarities[sorted_sim_index]

    def search(self, query: str, max_res=100):
        """Filter and preprocess query for better results and search with the model"""
        if not re.search(r't[^h\-\s]{1,6}bingen', query.lower()): # add tübingen if it is not part of the query
            query += ' tuebingen'
        filtered = [word for word in query.split() if word not in self.stop_words]# filter query for stop words
        filtered_query = ' '.join(filtered)
        urls, similarities = self.retrieve(filtered_query)

        relevant_urls = urls[:max_res]
        relevant_similarities = similarities[:max_res]

        # rerank
        query_embedding = self.model.embed(query)
        ranking_scores = []
        for relevant_url in tqdm(relevant_urls, "Reranking"):
            document_embedding = self.embedding_dict[relevant_url]
            ranking_score = self.model.resolve(query_embedding, document_embedding)
            ranking_scores.append(ranking_score)
        ranking_scores = np.array(ranking_scores)
        reranked_idxs = np.argsort(ranking_scores)

        ranked_urls = relevant_urls[reranked_idxs]

        ranked_documents = [self.docs.documents[url] for url in ranked_urls]

        # get sentence wise similarities
        sentence_wise_similarities = self.get_sentence_wise_similarities(query_embedding, relevant_urls)
        return ranked_urls, ranked_documents, ranking_scores[reranked_idxs], sentence_wise_similarities


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
