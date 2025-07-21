import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file
from transformers.models.clip.modeling_clip import clip_loss
import nltk
import math
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm 
from bs4 import BeautifulSoup, SoupStrainer
import json
from readability import Document as ReadabilityDocument

device = "cuda" if torch.cuda.is_available() else "cpu"

RELEVANT_TAGS = [
    "p", "h1", "h2", "h3", "h4", "h5", "h6"
]

class SiglipStyleModel(nn.Module):
    def __init__(self, model_name: str = "prajjwal1/bert-mini", loss_type: str = "siglip"):
        super().__init__()
        self.model_name = model_name
        self.loss_type = loss_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.bias = nn.Parameter(torch.zeros(1))
        self.to(device)

    # dummy return to keep this model compatible with search.py, which calls sentence_sim
    def sentence_sim(self, doc_tokens, query_tokens):
        return torch.zeros((1,doc_tokens)).to(device)

    def tokenize(self, texts: str | list[str]):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)

    def forward(self, query: str | list[str], answer: str | list[str], return_loss: bool = True):
        out_query = self.embed(query)
        out_answ = self.embed(answer)
        logits = out_query @ out_answ.t()  # + self.bias
        if return_loss:
            match self.loss_type:
                case "siglip": loss = self.siglip_loss(logits)
                case "clip": loss = clip_loss(logits)
        else: loss = None
        return {"loss": loss, "logits": logits}

    def siglip_loss(self, logits: torch.Tensor):
        sim = logits + self.bias
        eye = torch.eye(sim.size(0), device=sim.device)
        y = -torch.ones_like(sim) + 2 * eye
        loglik = F.logsigmoid(y * sim)
        nll = -torch.sum(loglik, dim=-1)
        loss = nll.mean()
        return loss

    def load(self, path: str):
        state_dict = load_file(path, device)
        self.load_state_dict(state_dict, strict=False)
        return self

    def embed(self, text, query=True):
        tokens = self.tokenize(text)
        out_text = self.encoder(**tokens).pooler_output
        out_embed = out_text / out_text.norm(p=2, dim=-1, keepdim=True)
        return out_embed

    def resolve(self, query_embeddings, document_embeddings):
        return query_embeddings @ document_embeddings.t()


class ColSentenceModel(nn.Module):
    def __init__(self, model_name="prajjwal1/bert-mini", embed_size=128, loss_type="siglip", use_max_sim=True):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.token_mapper = nn.Linear(self.bert_model.pooler.dense.out_features, embed_size)
        self.loss_type = loss_type
        nltk.download('punkt_tab')  # install the sentence level tokenizer
        self.use_max_sim = use_max_sim
        self.to(device)
    """
    def loss_function(self, predictions, gt_labels):
        true_labels_mask = gt_labels.bool()
        false_labels_mask = torch.logical_not(true_labels_mask)
        return - torch.sum(torch.log(predictions[true_labels_mask])) - torch.sum(torch.log(torch.ones_like(predictions[false_labels_mask])-predictions[false_labels_mask]))
    """
    # Intended tensor shapes:
    # doc_tokens: (batch_size, doc tokens (num sentences) -> may have padding, embedding size)
    # query_tokens: (batch_size, embedding size, query tokens (num sentences) -> may have padding)

    def sentence_sim(self, doc_tokens, query_tokens):
        query_tokens, doc_tokens, desired_query_shape, desired_doc_shape = self.sim_preprocess(query_tokens=query_tokens, doc_tokens=doc_tokens)
        return torch.sum(torch.bmm(doc_tokens, query_tokens), dim=2).reshape((desired_query_shape[0], desired_query_shape[1], desired_doc_shape[2]))

    def max_sim(self, doc_tokens, query_tokens):
        query_tokens, doc_tokens, desired_query_shape, _ = self.sim_preprocess(query_tokens=query_tokens, doc_tokens=doc_tokens)
        # shape after bmm (batch size, #doc_tokens, #query_tokens)
        return torch.sum(torch.max(torch.bmm(doc_tokens, query_tokens), dim=1, keepdim=True)[0], dim=2).reshape((desired_query_shape[0], desired_query_shape[1]))

    def max_avg_sim(self, doc_tokens, query_tokens, n: int):
        query_tokens, doc_tokens, desired_query_shape, _ = self.sim_preprocess(query_tokens=query_tokens, doc_tokens=doc_tokens)
        # shape after bmm (batch size, #doc_tokens, #query_tokens)
        best_n = torch.sort(torch.bmm(doc_tokens, query_tokens), dim=1, descending=True)[0][:, :int(len(doc_tokens)*(n/100)), :]
        return torch.sum(torch.mean(best_n, dim=1, keepdim=True), dim=2).reshape((desired_query_shape[0], desired_query_shape[1]))

    def avg_sim(self, doc_tokens, query_tokens):
        query_tokens, doc_tokens, desired_query_shape, _ = self.sim_preprocess(query_tokens=query_tokens, doc_tokens=doc_tokens)
        # shape after bmm (batch size, #doc_tokens, #query_tokens)
        return torch.sum(torch.mean(torch.bmm(doc_tokens, query_tokens), dim=1, keepdim=True), dim=2).reshape((desired_query_shape[0], desired_query_shape[1]))

    def restore_shape(self, tokens):
        if len(tokens.shape) == 2:
            return tokens.unsqueeze(0)
        elif len(tokens.shape) == 1:
            return tokens.unsqueeze(0).unsqueeze(0)
        else:
            return tokens

    def sim_preprocess(self, doc_tokens, query_tokens):
        # Remake shape to be 3d. 
        query_tokens = self.restore_shape(query_tokens)
        doc_tokens = self.restore_shape(doc_tokens)

        # move tensors to device
        query_tokens = query_tokens.to(device)
        doc_tokens = doc_tokens.to(device)

        # bring query token into expected shape
        query_tokens = query_tokens.transpose(1, 2) 
        desired_query_shape = (query_tokens.shape[0], doc_tokens.shape[0],
                                query_tokens.shape[1], query_tokens.shape[2])
        desired_doc_shape = (query_tokens.shape[0], doc_tokens.shape[0], 
                             doc_tokens.shape[1], doc_tokens.shape[2])
        query_tokens = query_tokens.unsqueeze(1).expand(desired_query_shape).flatten(0, 1)
        doc_tokens = doc_tokens.unsqueeze(0).expand(desired_doc_shape).flatten(0, 1)
        return query_tokens, doc_tokens, desired_query_shape, desired_doc_shape

    def resolve(self, query_embeddings, document_embeddings, max_sim=True):
        if max_sim:
            return self.max_avg_sim(document_embeddings, query_embeddings, 10)
        else:
            return self.avg_sim(document_embeddings, query_embeddings)

    def embed(self, text, batch_size=100, detach_results = False):  # sentences structure: (batch x sentences) embeddings
        if isinstance(text, str):
            text = [text] # wrap, because we expect something batch
        sentences, idx_map = self.extract_sentences(text)
        max_idx = 0
        embedding_tensors = []
        while max_idx < len(sentences):
            min_idx = max_idx
            max_idx = min(max_idx+batch_size,len(sentences))
            batch_sentences = sentences[min_idx:max_idx]
            tokens = self.tokenizer(batch_sentences, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512).to(device)
            embeddings = self.bert_model(**tokens).pooler_output
            final_embeddings = torch.nn.functional.normalize(self.token_mapper(embeddings), dim=1)
            if detach_results:
                final_embeddings = final_embeddings.detach().cpu()
            embedding_tensors.append(final_embeddings)
            torch.cuda.empty_cache() # embedding tensors are not freed up, but the model itself may be able to free up some space
        final_embeddings = torch.cat(embedding_tensors, dim=0)
        return torch.nn.utils.rnn.pad_sequence([final_embeddings[start:end] for start, end in idx_map], batch_first=True)

    def extract_sentences(self, texts):
        sentence_lists = []
        idx_map = []
        low_idx = 0
        for text in texts:
            sentences = nltk.tokenize.sent_tokenize(text, language='english')
            sentence_lists.extend(sentences)
            idx_map.append((low_idx, low_idx + len(sentences)))
            low_idx = low_idx + len(sentences)
        return sentence_lists, idx_map

    def forward(self, query, answer, return_loss=True, sentence_wise=False):
        query_embeddings = self.embed(query)
        document_embeddings = self.embed(answer)
        if self.use_max_sim:
            if sentence_wise:
                logits = self.sentence_sim(document_embeddings, query_embeddings)
            else:
                logits = self.max_sim(document_embeddings, query_embeddings)
        else:
            logits = self.avg_sim(document_embeddings, query_embeddings)
        if return_loss:
            match self.loss_type:
                case "siglip": loss = self.siglip_loss(logits)
                case "clip": loss = clip_loss(logits)
        else: loss = None
        return {"loss": loss, "logits": logits}

    def siglip_loss(self, logits):
        sim = logits + self.bias
        eye = torch.eye(sim.size(0), device=sim.device)
        y = -torch.ones_like(sim) + 2 * eye
        loglik = F.logsigmoid(y * sim)
        nll = -torch.sum(loglik, dim=-1)
        loss = nll.mean()
        return loss

    def load(self, path):
        state_dict = load_file(path, device)
        self.load_state_dict(state_dict, strict=False)
        return self


class BM25():
    def __init__(self, b=0.75, k=1.2):
        self.b = b
        self.k = k

    # RELEVANT_TAGS = [
    #     "p", "h1", "h2", "h3", "h4", "h5", "h6",
    # ]

    # def preprocess_html(self, html: str, seperator: str = '. ') -> str:
    #     relevant_tags = SoupStrainer(RELEVANT_TAGS)
    #     soup = BeautifulSoup(html, 'html.parser', parse_only=relevant_tags)
    #     text = soup.get_text(separator=seperator, strip=True)
    #     return text.strip().lower()

    def preprocess_html(self, html: str, seperator: str = ' ') -> str:
        readable_doc = ReadabilityDocument(html)
        summary_html = readable_doc.summary(html_partial=True) if readable_doc else None
        summary_text = None
        if summary_html:
            summary_soup = BeautifulSoup(summary_html, 'lxml')
            for a_tag in summary_soup.find_all('a'):
                a_tag.decompose()
            paragraphs = summary_soup.find_all('p')
            for paragraph in paragraphs:
                summary_text = "".join(paragraph.get_text(separator=seperator, strip=True))

        if summary_text:
            text = summary_text
        else:
            whole_soup = BeautifulSoup(html, 'lxml')
            whole_text = whole_soup.get_text(separator=seperator, strip=True)
            text = whole_text
        return text.strip().lower()

    def bag_words(self, words: list[str], doc_freqs: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
        word_bag = {}
        for word in words:
            if word in word_bag.keys():
                word_bag[word] += 1
            else:
                word_bag[word] = 1
                if word in doc_freqs.keys():
                    doc_freqs[word] += 1
                else:
                    doc_freqs[word] = 1
        return word_bag, doc_freqs

    def bag_documents(self, documents: dict[str, str]) -> tuple[dict[str, dict[str, int]], dict[str, float], float, dict[str, int]]:
        word_bags = {}
        doc_lengths = {}
        doc_freqs = {}
        for url in tqdm(documents.keys(), "preprocessing"):
            document = self.preprocess_html(documents[url])
            words = nltk.tokenize.word_tokenize(document)
            words = [re.sub(r'[^\w\s]', '', word)
                     for word in words if re.sub(r'[^\w\s]', '', word)]  # regex courtesy of G4G
            doc_lengths[url] = len(words)
            word_bag, doc_freqs = self.bag_words(words, doc_freqs)
            word_bags[url] = word_bag
        idfs = self.calc_idf(doc_freqs, len(doc_lengths.keys()))
        if len(doc_lengths) > 0:
            avgdl = sum(doc_lengths.values()) / len(doc_lengths.values())
        else:
            avgdl = 0
        return word_bags, idfs, avgdl, doc_lengths

    def calc_idf(self, doc_freqs: dict[str, int], num_docs: int) -> dict[str, float]:
        idfs = {}
        for term in doc_freqs.keys():
            idfs[term] = math.log((num_docs - doc_freqs[term] + 0.5) / (doc_freqs[term] + 0.5) +1)
        return idfs

    def preprocess(self, documents: dict[str, str]) -> None:
        #self.documents = documents  # order of documents determines the implicit indexing
        word_bags, idfs, avgdl, doc_lenghts = self.bag_documents(documents) # documents 
        self.word_bags = word_bags
        self.idfs = idfs
        self.avgdl = avgdl
        self.doc_lengths = doc_lenghts

    def calculate_rels(self, query: str) -> np.ndarray:
        query_terms = nltk.tokenize.word_tokenize(query)
        relevance_list = []
        for doc_idx in range(len(self.word_bags.keys())):
            relevance_list.append(self.get_relevance(query_terms, doc_idx))
        return np.array(relevance_list)

    def get_relevance(self, query_terms: list[str], doc_url: str) -> float:
        score = 0
        for query_term in query_terms:
            if query_term in self.idfs.keys():
                idfs_score = self.idfs[query_term]
            else:
                idfs_score = math.log((len(self.word_bags.keys()) + 0.5)/0.5 +1)

            if query_term in self.word_bags[doc_url].keys():
                frequency = self.word_bags[doc_url][query_term]/self.doc_lengths[doc_url]
            else:
                frequency = 0
            
            doc_len_ratio = self.doc_lengths[doc_url]/self.avgdl
            numerator = (frequency * (self.k + 1))
            denominator = (frequency + self.k * (1-self.b + self.b*doc_len_ratio))
            
            score += idfs_score * numerator/denominator
        return score

    def resolve(self, query: str) -> dict[str, float]:
        query_terms = nltk.tokenize.word_tokenize(query.lower())
        relevancies = {}
        for doc_url in tqdm(self.word_bags.keys(), "Retrieving"):
            relevancies[doc_url] = self.get_relevance(query_terms, doc_url)
        return relevancies
    
    def save(self, path=""):
        state_dict = {
            "word_bags": self.word_bags,
            "idfs": self.idfs,
            "avgdl": self.avgdl,
            "doc_lengths": self.doc_lengths,
            "k": self.k,
            "b": self.b
        }
        with open(path, "w") as f:
            json.dump(state_dict, f)
        
    def load(self, path=""):
        with open(path, "r") as f:
            state_dict = json.load(f)
        self.word_bags = state_dict["word_bags"]
        self.idfs = state_dict["idfs"]
        self.avgdl = state_dict["avgdl"]
        self.doc_lengths = state_dict["doc_lengths"]
        self.k = state_dict["k"]
        self.b = state_dict["b"]
        return self
        

class MentorModel(nn.Module):
    def __init__(self):
        model = SentenceTransformer("infly/inf-retriever-v1", trust_remote_code=True)
        model.max_seq_length = 8192
        self.to(device)

    def forward(self, query: str | list[str], answer: str | list[str], return_loss: bool = True):
        out_query = self.embed(query)
        out_answ = self.embed(answer)
        logits = out_query @ out_answ.t()  # + self.bias
        if return_loss:
            match self.loss_type:
                case "siglip": loss = self.siglip_loss(logits)
                case "clip": loss = clip_loss(logits)
        else: loss = None
        return {"loss": loss, "logits": logits}

    def siglip_loss(self, logits: torch.Tensor):
        sim = logits + self.bias
        eye = torch.eye(sim.size(0), device=sim.device)
        y = -torch.ones_like(sim) + 2 * eye
        loglik = F.logsigmoid(y * sim)
        nll = -torch.sum(loglik, dim=-1)
        loss = nll.mean()
        return loss

    def load(self, path: str):
        state_dict = load_file(path, device)
        self.load_state_dict(state_dict, strict=False)
        return self

    def embed(self, text, query=True):
        if query:
            out_embed = self.model.encode(text, prompt_name="query")
        else:
            out_embed = self.model.encode(text)
        return out_embed

    def resolve(self, query_embeddings, document_embeddings):
        return query_embeddings @ document_embeddings.t()