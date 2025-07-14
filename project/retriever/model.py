import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file
from transformers.models.clip.modeling_clip import clip_loss

device = "cuda" if torch.cuda.is_available() else "cpu"

class RetrieverModel(nn.Module):
    def __init__(self, model_name="prajjwal1/bert-mini", loss_type="siglip"):
        super().__init__()
        self.model_name = model_name
        self.loss_type = loss_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.bias = nn.Parameter(torch.zeros(1))
        self.to(device)

    def tokenize(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)

    def forward(self, query, answer, return_loss=True):
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