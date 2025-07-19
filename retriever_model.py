import sklearn as sk
from torch import nn
from collections import OrderedDict
from transformers import BertTokenizer, BertModel
import torch
import nltk

class BertRetriever(): # Idea: bastardization of ColBert approach
    def __init__(self, bert_model_name="bert-base-uncased", embed_size=256):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.token_mapper = nn.Linear(756, embed_size)
    
    def loss_function(self, predictions, gt_labels):
        true_labels_mask = gt_labels.bool()
        false_labels_mask = torch.logical_not(true_labels_mask)
        return - torch.sum(torch.log(predictions[true_labels_mask])) - torch.sum(torch.log(torch.ones_like(predictions[false_labels_mask])-predictions[false_labels_mask]))

    # Intended tensor shapes: 
    # doc_tokens: (batch_size, doc tokens (num sentences) -> may have padding, embedding size)
    # query_tokens: (batch_size, embedding size, query tokens (num sentences) -> may have padding)
    def max_sim(self, doc_tokens, query_tokens, sentence_wise):
        if sentence_wise:
            return torch.sum(torch.bmm(doc_tokens, query_tokens), dim=2)
        else:
            return torch.sum(torch.max(torch.bmm(doc_tokens, query_tokens), dim=1), dim=2) # shape after bmm (batch size, #doc_tokens, #query_tokens)

    def embed(self, sentences, idx_map): # sentences structure: (batch x sentences) embeddings
        tokens = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        embeddings = self.bert_model(**tokens).pooler_output
        final_embeddings = self.token_mapper(embeddings)
        return torch.nn.utils.rnn.pad_sequence([final_embeddings[start:end] for start, end in idx_map], batch_first=True)
    
    def extract_sentences(self, texts):
        sentence_lists = []
        idx_map = []
        low_idx = 0
        for text in texts:
            sentences = nltk.tokenize.sent_tokenize(text, language='english')
            sentence_lists.append(sentences)
            idx_map.append((low_idx, low_idx+len(sentences)))
            low_idx = len(sentences)
        return sentence_lists, idx_map
    
    def predict_relevance(self, documents, queries, sentence_wise=False):
        query_sentences, query_idx_map = self.extract_sentences(queries)
        document_sentences, document_idx_map = self.extract_sentences(documents)
        query_embeddings = self.embed(query_sentences, query_idx_map)
        query_embeddings = torch.transpose(query_embeddings, 1, 2)
        document_embeddings = self.embed(document_sentences, document_idx_map)
        relevance_predictions = self.max_sim(document_embeddings, query_embeddings, sentence_wise)
        return relevance_predictions