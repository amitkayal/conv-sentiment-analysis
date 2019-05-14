import pandas as pd
import os
import torch
from collections import defaultdict
torch.manual_seed(42)


PAD_INDEX = 1

# TODO implement padding
class StanfordDatasetReader():
    def __init__(self, sst_dir, split_idx):
        merged_dataset = self.get_merged_dataset(sst_dir)
        self.dataset = merged_dataset[merged_dataset["splitset_label"] == split_idx]

    def get_merged_dataset(self, sst_dir):
        sentiment_labels = pd.read_csv(os.path.join(sst_dir, "sentiment_labels.txt"), sep="|")
        sentence_ids = pd.read_csv(os.path.join(sst_dir, "datasetSentences.txt"), sep="\t")
        dictionary = pd.read_csv(os.path.join(sst_dir, "dictionary.txt"), sep="|", names=['phrase', 'phrase ids'])
        train_test_split = pd.read_csv(os.path.join(sst_dir, "datasetSplit.txt"))
        sentence_phrase_merge = pd.merge(sentence_ids, dictionary, left_on='sentence', right_on='phrase')
        sentence_phrase_split = pd.merge(sentence_phrase_merge, train_test_split, on='sentence_index')
        return pd.merge(sentence_phrase_split, sentiment_labels, on='phrase ids').sample(frac=1)

    def discretize_label(self, label):
        if label <= 0.2: return 0
        if label <= 0.4: return 1
        if label <= 0.6: return 2
        if label <= 0.8: return 3
        return 4

    def __len__(self):
        return self.dataset.shape[0]

    def word_to_index(self, word):
        if word in self.w2i:
            return self.w2i[word]
        else:
            return self.w2i["<OOV>"]

    def __getitem__(self, idx):
        return {"sentence": [self.word_to_index(x) for x in self.dataset.iloc[idx, 1].split()],
                "label": self.discretize_label(self.dataset.iloc[idx, 5])}
    
    def set_w2i(self, w2i):
        self.w2i = w2i

    
def safe_collate(kernel_size):
    def collate_fn(batch):
        max_sample_length = max(max(len(x['sentence']) for x in batch), kernel_size)
        batch_sentence = []
        batch_labels = []
        real_length = []
        for i in range(len(batch)):
            batch_sentence.append(batch[i]['sentence'] + [PAD_INDEX] * (max_sample_length - len(batch[i]['sentence'])))
            batch_labels.append(batch[i]['label'])
            real_length.append(len(batch[i]))
        return {'sentence': torch.LongTensor(batch_sentence), 'label':  torch.LongTensor(batch_labels), 'length': real_length}
    return collate_fn

def build_vocab(sentences):
    w2i = defaultdict(lambda : len(w2i))
    w2i["<PAD>"] += 1
    w2i["<OOV>"] += 1
    w2i["<UNK>"] += 1
    sentences = list(sentences)
    for s in sentences:
        words = s.split()
        for w in words:
            if w not in w2i:
                w2i[w] += 1    
    return w2i