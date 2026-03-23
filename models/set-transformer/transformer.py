import torch
import torch.nn as nn

class SetTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim,hidden_dim,pretrained_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(pretrained_weights)
            self.embedding.weight.requires_grad = True
        
        