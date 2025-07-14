import math
import torch

# assume train dataset has the structure (query_id, document_id, relevant/not_relevant) for each row. (shape is dataset_size x 3)
# assume test dataset has the structure (query_id, document_id, relevance_score)
class BertFinetuningPipeline():
    def __init__(self, train_dataset, test_dataset, documents, queries, model, lossfunc, epochs, batch_size, optimizer):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.lossfunc = lossfunc
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer

        # identifier of the corpus with regards to which we load the id
        self.documents = documents
        self.queries = queries

    # TODO: regular checkpointing
    def train(self):
        for epoch_idx in range(0, self.epochs):
            print(f"EPOCH {epoch_idx}")
            batch_start = 0
            while batch_start < len(self.train_dataset):
                batch_end = min(batch_end+self.batch_size)
                batch = self.train_dataset[batch_start:batch_end]
                gt_labels = batch[:,2]
                documents = self.documents.get_text(batch[:,1])
                queries = self.documents.get_text(batch[:,0])
                predictions = self.model.predict_relevance(documents, queries)
                loss = self.lossfunc(predictions, gt_labels)
                loss.backward()
                self.optimizer.step()

    # TODO: calculate and report NDCG with the gt relevancies
    def test(self):
        ndcgs = []
        discount_vector = torch.tensor([1/math.log(idx+1) for idx in range(0,100)])
        for start_idx in range(0, len(self.test_dataset), 100): # load in chunks of 100, since rankings are in chunks of 100 per query. Careful, if something fucks up the alignment, this breaks
            batch = self.test_dataset[start_idx:start_idx+100] # no failsafes, the lenght of the training dataset has to be a multiple of 100
            query = self.get_text(batch[0][0]) # can get query from the first.
            documents = self.get_text(batch[:][1])
            relevancies = batch[:][2]
            ndcg = 


