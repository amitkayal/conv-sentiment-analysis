# Convolutional Neural Networks for Sentence Classification

This is an implementation of the [paper], Convolutional Neural Networks for Sentence Classification by Yoon Kim. The network is trained on stanford sentiment treebank and achieves 43.34 accuracy on the 5 class stanford sentiment dataset.

# Requirements

  - torch 1.0.1
  - pytorch-ignite 0.2.0
  - fire 0.1.3
  - Python 3.6

# Demo
```sh
python demo.py
Please input a sentence
This is good
Very Bad: 0.04406440258026123
Bad: 0.08807741850614548
Neutral: 0.1189938336610794
Good: 0.3565130829811096
Very Good: 0.39235129952430725
```

# Training
```sh
python run.py train  --sst-dir ../dataset/stanfordSentimentTreebank/   --model-save-dir ../checkpoint --batch-size 32  --kernel-sizes [2,5,6] --stride 1, --num-filters 200 --dropout-prob 0.5 --n-classes 5 --embedding-file ../Wordvectors/word2vec/GoogleNews-vectors-negative300.bin --embedding-dim  300 --learning-rate 0.1 --num-epochs 100 --patience 20 --weight-decay 0.001
```
Please download the training data from http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
# Testing
```sh
python run.py  test  --sst-dir ../dataset/stanfordSentimentTreebank/ --model-path ../checkpoint/emb_d_300_nameGoogle_num_filters_200_kernel_sizes\=2_5_6_l2_0.001_drp_0.5/test_trainer_mymodel_16_validation_loss\=0.408046.pth  --batch-size 32  --kernel-sizes [2,5,6] --stride 1, --num-filters 200 --dropout-prob 0.5 --n-classes 5  --embedding-dim 300  --vocab-path ../checkpoint/emb_d_300_nameGoogle_num_filters_200_kernel_sizes\=2_5_6_l2_0.001_drp_0.5/vocab.pkl
```
[paper]: https://www.aclweb.org/anthology/D14-1181.pdf
# License
MIT
