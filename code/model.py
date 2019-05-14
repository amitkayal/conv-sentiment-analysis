import torch.nn as nn
import torch
from embedding import WordEmbedder
from torch.nn import Conv2d

torch.manual_seed(42)

class CNNClassifier(nn.Module):
    def __init__(self, kernel_sizes, stride, num_filters, dropout_prob, n_classes, w2i, embedding_dim, embedding_file=None):
        super(CNNClassifier, self).__init__()
        self.n_conv_layers = len(kernel_sizes) * num_filters
        self.embedder = WordEmbedder(embedding_file, embedding_dim, w2i)
        self.conv2d = nn.ModuleList([nn.Conv2d(1, num_filters, (s, embedding_dim), stride) for s in kernel_sizes])
        [torch.nn.init.xavier_uniform_(conv.weight) for conv in self.conv2d]
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(self.n_conv_layers, n_classes)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        
    def forward(self, data):
        embeddings = self.embedder(data)
        embeddings = embeddings.unsqueeze(1)
        conv_outs = []
        for conv in self.conv2d:
            conv_outs.append(torch.max(
                torch.nn.functional.relu(
                    conv(embeddings)
                ), dim=2
                )[0].squeeze(-1))
        concatenated_convs = torch.cat(conv_outs, dim=1)
        dropouts = self.dropout(concatenated_convs)
        logits = self.linear(dropouts)
        return logits
