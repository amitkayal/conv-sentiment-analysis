import torch
import numpy as np
from torch import nn
from gensim.models import KeyedVectors
torch.manual_seed(42)

def to_torch(numpy_array):
     return torch.from_numpy(numpy_array)

class WordEmbedder(nn.Module):
    def __init__(self, embedding_file, embedding_dim, w2i):
        super(WordEmbedder, self).__init__()
        self.word2Index = w2i
        self.embedding_dim = embedding_dim
        if embedding_file:
            self.embedding_matrix, absent_embeddings = self.read_embeddding(embedding_file)
            self.embedding = nn.Embedding.from_pretrained(to_torch(self.embedding_matrix.astype(np.float32)), freeze=True)
        else:
            self.embedding = nn.Embedding(len(w2i) + 1, embedding_dim)

    def read_embeddding(self, embedding_file):
        if ".bin"  in embedding_file:
            print("Reading binary embeddings")
            embedding_dict = self.read_binary_embeddings(embedding_file)
        else:
            print("Read text embeddings")
            embedding_dict = self.read_text_embeddings(embedding_file)


        embedding_matrix = np.zeros((len(self.word2Index) + 1, self.embedding_dim), dtype=np.float32)
        absent_embeddings = []
        for word, index in self.word2Index.items():
            vector = embedding_dict.get(word)
            if vector is not None:
                embedding_matrix[index, :] = vector
            else:
                embedding_matrix[index, :] = np.random.uniform(size=(1, self.embedding_dim))
                absent_embeddings.append(word)
        return embedding_matrix, absent_embeddings

    def read_binary_embeddings(self, embedding_file):
        embeddings_dict = {}
        wv_from_bin = KeyedVectors.load_word2vec_format(embedding_file, binary=True) 
        for word, vector in zip(wv_from_bin.vocab, wv_from_bin.vectors):
            coefs = np.asarray(vector, dtype='float32')
            embeddings_dict[word] = (coefs / np.linalg.norm(coefs))
        return embeddings_dict

    def read_text_embeddings(self, embedding_file):
        embedding_dict, absent_embeddings = {}, []
        with open(embedding_file, "r") as f:
            for line in f:
                values = line.split()
                word = values[0]
                word_embedding = np.asarray(values[1:], dtype=np.float32)
                word_embedding = word_embedding
                if np.sum(word_embedding) == 0:
                    print(f"Assigning random embedding to word {word}")
                    word_embedding = np.random.uniform(size=(1, self.embedding_dim))
                embedding_dict[word] = word_embedding / np.linalg.norm(word_embedding)
        return embedding_dict

    def get_output_dim(self):
        return self.embedding_dim

    def forward(self, vectors):
        return self.embedding(vectors)