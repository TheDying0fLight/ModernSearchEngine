from torch import nn
from collections import OrderedDict
from transformers import BertTokenizer, BertModel

class StaticBertEmbedder():
    def __init__(self, hidden_layers=3, bert_model_name=None):
        architecture = OrderedDict()
        idx=0
        while idx < hidden_layers*2:
            architecture[idx] = nn.Linear(768, 768)
            architecture[idx+1] = nn.ELU()
            idx+=2

        self.mapping_function = nn.Sequential(architecture)
        self.loss_function = nn.TripletMarginLoss()

        if bert_model_name is None:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        else:
            self.bert_model = BertModel.from_pretrained(bert_model_name)
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    # this fit uses triplet loss, which is based on euclidean distance. Cosine similarity is usually better for what we are trying to achieve however,
    # so it may be better to use the CosineEmbeddingLoss. this requires different parameters
    def fit(self, relevance_triplets, batch_size=100, epochs=3): # [query, relevant, irrelevant] structure
        for epoch in range(0, epochs):
            print(epoch)
            for batch_idx in range(0, batch_size):
                batch_start = batch_idx*batch_size
                batch_end = min((batch_idx+1)*batch_size,relevance_triplets.shape[1])
                query_embeddings = self.mapping_function(relevance_triplets[0,batch_start:batch_end,:])
                relevant_embeddings = self.mapping_function(relevance_triplets[1,batch_start:batch_end,:])
                irrelevant_embeddings = self.mapping_function(relevance_triplets[2,batch_start:batch_end,:])
                loss = self.loss_function(query_embeddings, relevant_embeddings, irrelevant_embeddings)
                loss.backward()

    def get_embedding(self, query):
        tokens = self.tokenizer(query)
        bert_embedding = self.bert_model(tokens)
        final_embedding = self.mapping_function(bert_embedding)
        return final_embedding

class Word2VecEmbedder(): #getting ahold of word2vec weights seems surprisingly tricky... best shot is currently gensim + https://drive.google.com/file/d/1ETEzH8X7uM_xXtIEuNLgz9VL7eQEeE_V/view
    pass

class MpnetEmbedder(): # all-mpnet-base-v2 seems like exactly the type of embedder we need. However is this too close to a model specifically for retrieval? Its explicitly made for semantic search among other things.
    pass

