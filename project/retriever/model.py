import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file
from transformers.models.clip.modeling_clip import clip_loss
import nltk
import math
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


class SiglipStyleModel(nn.Module):
    def __init__(self, model_name: str = "prajjwal1/bert-mini", loss_type: str = "siglip"):
        super().__init__()
        self.model_name = model_name
        self.loss_type = loss_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.bias = nn.Parameter(torch.zeros(1))
        self.to(device)

    def tokenize(self, texts: str | list[str]):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)

    def forward(self, query: str | list[str], answer: str | list[str], return_loss: bool = True):
        tok_query = self.tokenize(query)
        tok_answ = self.tokenize(answer)
        out_query = self.encoder(**tok_query).pooler_output
        out_answ = self.encoder(**tok_answ).pooler_output
        out_query = out_query / out_query.norm(p=2, dim=-1, keepdim=True)
        out_answ = out_answ / out_answ.norm(p=2, dim=-1, keepdim=True)
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
    def __init__(self, model_name: str = "prajjwal1/bert-mini", embed_size: str = 128, loss_type: str = "siglip"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.token_mapper = nn.Linear(self.bert_model.pooler.dense.out_features, embed_size)
        self.loss_type = loss_type
        nltk.download('punkt_tab')  # install the sentence level tokenizer
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

    def max_sim(self, doc_tokens, query_tokens, sentence_wise, full_mat=True):
        if full_mat:
            desired_query_shape = (query_tokens.shape[0], doc_tokens.shape[0],
                                   query_tokens.shape[1], query_tokens.shape[2])
            desired_doc_shape = (query_tokens.shape[0], doc_tokens.shape[0], doc_tokens.shape[1], doc_tokens.shape[2])
            query_tokens = query_tokens.unsqueeze(1).expand(desired_query_shape).flatten(0, 1)
            doc_tokens = doc_tokens.unsqueeze(0).expand(desired_doc_shape).flatten(0, 1)
            if sentence_wise:
                return torch.sum(torch.bmm(doc_tokens, query_tokens), dim=2).reshape((desired_query_shape[0], desired_query_shape[1], desired_doc_shape[2]))
            else:
                # shape after bmm (batch size, #doc_tokens, #query_tokens)
                return torch.sum(torch.max(torch.bmm(doc_tokens, query_tokens), dim=1, keepdim=True)[0], dim=2).reshape((desired_query_shape[0], desired_query_shape[1]))
        else:
            if sentence_wise:
                return torch.sum(torch.bmm(doc_tokens, query_tokens), dim=2)
            else:
                # shape after bmm (batch size, #doc_tokens, #query_tokens)
                return torch.sum(torch.max(torch.bmm(doc_tokens, query_tokens), dim=1, keepdim=True)[0], dim=2)

    def resolve(self, query_embeddings, document_embeddings):
        return self.max_sim(document_embeddings, query_embeddings, sentence_wise=False)

    def embed(self, text, query=True):  # sentences structure: (batch x sentences) embeddings
        sentences, idx_map = self.extract_sentences(text)
        tokens = self.tokenizer(sentences, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).to(device)
        embeddings = self.bert_model(**tokens).pooler_output
        final_embeddings = torch.nn.functional.normalize(self.token_mapper(embeddings), dim=1)
        if query:
            return torch.nn.utils.rnn.pad_sequence([final_embeddings[start:end] for start, end in idx_map], batch_first=True).transpose(1, 2)
        else:
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
        query_embeddings = self.embed(query, query=True)
        document_embeddings = self.embed(answer, query=False)
        logits = self.max_sim(document_embeddings, query_embeddings, sentence_wise)
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

    def bag_words(self, words, doc_freqs):
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

    def bag_documents(self, documents):
        word_bags = []
        doc_lengths = []
        doc_freqs = {}
        for document in documents:
            words = nltk.tokenize.word_tokenize(document)
            doc_lengths.append(len(words))
            word_bag, doc_freqs = self.bag_words(words, doc_freqs)
            word_bags.append(word_bag)
        idfs = self.calc_idf(doc_freqs, len(doc_lengths))
        if doc_lengths > 0:
            avgdl = sum(doc_lengths) / len(doc_lengths)
        else:
            avgdl = 0
        return word_bags, idfs, avgdl

    def calc_idf(self, doc_freqs, num_docs):
        idfs = {}
        for term in doc_freqs.keys():
            idfs[term] = math.log((num_docs - doc_freqs[term] + 0.5) / (doc_freqs[term] + 0.5) + 1)
        return idfs

    def preprocess(self, documents):
        self.documents = documents  # order of documents determines the implicit indexing
        word_bags, idfs, avgdl = self.bag_documents(documents)
        self.word_bags = word_bags
        self.idfs = idfs
        self.avgdl = avgdl

    def calculate_rels(self, query):
        query_terms = nltk.tokenize.word_tokenize(query)
        relevance_list = []
        for doc_idx in range(len(self.word_bags)):
            relevance_list.appen(self.get_relevance(query_terms, doc_idx))
        return np.array(relevance_list)

    def get_relevance(self, query_terms, doc_idx):
        score = 0
        for query_term in query_terms:
            score += self.idfs[query_term] * (self.word_bags[doc_idx][query_term] * (self.k + 1)) / (
                self.word_bags[doc_idx][query_term] + self.k * (1 - self.b + self.b * len(self.word_bags) / self.avgdl))
        return score
