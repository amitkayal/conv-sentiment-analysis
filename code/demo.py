import argparse
import torch
import pickle as pkl
import os
from model import CNNClassifier

torch.manual_seed(42)

dir_prefix = "../checkpoint/emb_d_300_nameGoogle_num_filters_200_kernel_sizes=2_5_6_l2_0.001_drp_0.5/"
vocab_path = os.path.join(dir_prefix, "vocab.pkl")

kernel_sizes = [2, 5, 6]
PAD_INDEX = 1
w2i = pkl.load(open(vocab_path, "rb"))
classifier_parameters = torch.load(os.path.join(
    dir_prefix, "test_trainer_mymodel_16_validation_loss=0.408046.pth"))
stride = 1
num_filters = 200
dropout_prob = 0.5
n_classes = 5
embedding_dim = 300
classifier = CNNClassifier(kernel_sizes=kernel_sizes, stride=stride,
                           num_filters=num_filters, dropout_prob=dropout_prob, n_classes=n_classes,
                           embedding_file=None, embedding_dim=embedding_dim, w2i=w2i)
classifier.load_state_dict(classifier_parameters)
classifier = classifier.eval()

labels = ["Very Bad", "Bad", "Neutral", "Good", "Very Good"]
while True:
    words = input("Please input a sentence\n").split()
    sample_length = max(len(words), max(kernel_sizes))
    tokens = [w2i["<OOV>"] if word not in w2i else w2i[word]
              for word in words] + [PAD_INDEX] * (sample_length - len(words))
    tokens = torch.LongTensor(tokens)
    logits = classifier(tokens.unsqueeze(0))
    probs = torch.nn.functional.softmax(logits, dim=1).data.numpy()[0]
    print("\n")
    for label, prob in zip(labels, probs):
        print(f"{label}: {prob}")
    print("\n")
