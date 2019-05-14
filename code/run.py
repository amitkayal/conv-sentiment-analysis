import fire
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle as pkl
import os
import shutil

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset_reader import StanfordDatasetReader, safe_collate, build_vocab
from model import CNNClassifier
from copy_model import CNN_Text


from ignite.contrib.handlers import ProgressBar

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import EarlyStopping

####### Commands ##########
# python train.py train  --sst-dir ../dataset/stanfordSentimentTreebank/   --model-save-dir /Users/talurj/Documents/Research/MyResearch/Learning/Neubig/assignment-1/checkpoint --batch-size 32  --kernel-sizes [3,4,5] --stride 1, --num-filters 200 --dropout-prob 0.5 --n-classes 5 --embedding-file ~/Documents/Research/MyResearch/SemEval/Emoconnect/Wordvectors/word2vec/GoogleNews-vectors-negative300.bin --embedding-dim  300 --learning-rate 0.1 --num-epochs 100 --patience 20 --weight-decay 0.001

torch.manual_seed(42)


def load_data(sst_dir, batch_size, kernel_size, w2i=None):
    """
    Load stanford sentiment treebank data
    """
    train_st_data = StanfordDatasetReader(sst_dir, 1)
    if w2i is None:
        w2i = build_vocab(train_st_data.dataset['sentence'])
    train_st_data.set_w2i(w2i)
    test_st_data = StanfordDatasetReader(sst_dir, 2)
    test_st_data.set_w2i(w2i)
    validation_st_data = StanfordDatasetReader(sst_dir, 3)
    validation_st_data.set_w2i(w2i)
    train_data_loader = DataLoader(
        train_st_data, batch_size=batch_size, shuffle=True, collate_fn=safe_collate(kernel_size))
    test_data_loader = DataLoader(
        test_st_data, batch_size=batch_size, shuffle=True, collate_fn=safe_collate(kernel_size))
    validation_data_loader = DataLoader(
        validation_st_data, batch_size=batch_size, shuffle=True, collate_fn=safe_collate(kernel_size))
    return train_data_loader, test_data_loader, validation_data_loader, w2i


def make_unique_dir_name(params):
    kernel_sizes = "_".join([str(x) for x in params["kernel_sizes"]])
    embedding_name = ""
    if "glove" in params["embedding_file"]:
        embedding_name = "glove"
    elif "google" in params["embedding_file"].lower():
        embedding_name = "Google"
    dirname = f"emb_d_{params['embedding_dim']}_name{embedding_name}_num_filters_{params['num_filters']}_kernel_sizes={kernel_sizes}_\
l2_{params['weight_decay']}_drp_{params['dropout_prob']}"
    return dirname


class Trainer():
    def train_model(self, train_data_loader, validation_data_loader, w2i, params):
        classifier = CNNClassifier(kernel_sizes=params["kernel_sizes"], stride=params["stride"],
                                   num_filters=params["num_filters"], dropout_prob=params[
                                       "dropout_prob"], n_classes=params["n_classes"],
                                   embedding_file=params["embedding_file"], embedding_dim=params["embedding_dim"], w2i=w2i)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adadelta(classifier.parameters(
        ), lr=0.95, weight_decay=params["weight_decay"])
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

        def step(engine, batch):
            sentence, labels = batch['sentence'], batch['label']
            classifier.train()
            classifier.zero_grad()
            logits = classifier(sentence[:, :])
            output = loss(logits, labels)
            output.backward()
            optimizer.step()
            return {
                'loss': output.item()
            }

        def eval_score_function(engine):
            classifier.eval()
            n_matches, total = 0, 0
            for batch in validation_data_loader:
                sentence, true_labels = batch['sentence'], batch['label']
                predictions = classifier(sentence[:, :])
                predicted_labels = np.argmax(predictions.data.numpy(), axis=1)
                n_matches += np.sum(true_labels.data.numpy()
                                    == predicted_labels)
                total += true_labels.shape[0]
            return n_matches/total

        def lr_schedule_handler(engine):
            val_loss = eval_score_function(engine)
            scheduler.step(val_loss)

        trainer = Engine(step)
        dir_name = make_unique_dir_name(params)
        try:
            shutil.rmtree(os.path.join(params["model_save_dir"], dir_name))
        except:
            pass
        os.makedirs(os.path.join(
            params["model_save_dir"], dir_name), exist_ok=True)

        checkpoint_handler = ModelCheckpoint(os.path.join(params["model_save_dir"], dir_name),
                                             "test_trainer", n_saved=10, require_empty=False, score_function=eval_score_function,
                                             score_name="validation_loss", save_as_state_dict=True)

        early_stopping_handler = EarlyStopping(
            patience=params["patience"], score_function=eval_score_function, trainer=trainer)
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, early_stopping_handler)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
                                  'mymodel': classifier})
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lr_schedule_handler)
        monitoring_metrics = ['loss']
        RunningAverage(alpha=0.9, output_transform=lambda x: x['loss']).attach(
            trainer, 'loss')
        pbar = ProgressBar()
        pbar.attach(trainer, metric_names=monitoring_metrics)
        trainer.run(train_data_loader, max_epochs=params["num_epochs"])

    def train(self, sst_dir, model_save_dir, batch_size, kernel_sizes, stride,
              num_filters, dropout_prob, n_classes, embedding_file, embedding_dim, learning_rate, num_epochs, patience, weight_decay):
        """
        Train the convolution neural network for sentiment classification on stanford sentiment treebank.

        Args:
            sst_dir (str): Path stanford sentiment treebank dataset.
            model_save_dir (str): Directory where the model can be saved
            batch_size (int): Batch size for training
            kernel_sizes (list): List of convolutional kernel sizes
            stride (int): Convolution strides
            num_filters (int): Number of convolution filters
            dropout_prob (float): Probability with which the neurons ouput is set to zero
            n_classes (int): Number of output classes. Since we are working with stanford sentiment treeback, set n_classes as 5.
            embedding_file (str): Path to the word embedding vectors. Currently it supports glove and word2vec
            embedding_dim (int): Dimension of the word embedding
            learning_rate (float): Learning rate used in adadelta
            num_epochs (int): Number of epochs to train on
            patience (int): Number of epochs to wait before early_stopping can be used. Please look at
                            https://pytorch.org/ignite/_modules/ignite/handlers/early_stopping.html
            weight_decay (float): L2 regularization weight
        """

        train_data_loader, test_data_loader, validation_data_loader, w2i = load_data(
            sst_dir, batch_size, max(kernel_sizes))
        params = {"kernel_sizes": kernel_sizes, "stride": stride, "num_filters": num_filters,
                  "dropout_prob": dropout_prob, "n_classes": n_classes, "embedding_file": embedding_file,
                  "embedding_dim": embedding_dim, "lr": learning_rate, "num_epochs": num_epochs,
                  "patience": patience, "model_save_dir": model_save_dir, "weight_decay": weight_decay}
        self.train_model(train_data_loader,
                         validation_data_loader, params=params, w2i=w2i)
        dir_name = make_unique_dir_name(params)
        # Save the vocab for use during testing
        pkl.dump(dict(w2i), open(os.path.join(
            model_save_dir, dir_name, "vocab.pkl"), "wb"))

    def test(self, model_path, sst_dir, batch_size, kernel_sizes, stride, num_filters, dropout_prob, n_classes,
             embedding_dim, vocab_path):
        """
        Evaluate model on test dataset

        Args:
            model_path (str): Path to model checkpoint folder
            sst_dir (str): Path stanford sentiment treebank dataset.
            batch_size (int): Batch size for training
            kernel_sizes (list): List of convolutional kernel sizes
            stride (int): Convolution strides
            num_filters (int): Number of convolution filters
            dropout_prob (float): Probability with which the neurons ouput is set to zero
            n_classes (int): Number of output classes. Since we are working with stanford sentiment treeback, set n_classes as 5.
            embedding_file (str): Path to the word embedding vectors. Currently it supports glove and word2vec
            embedding_dim (int): Dimension of the word embedding
            vocab_path (str): Path to the vocabulary pickle file
        """
        classifier_parameters = torch.load(model_path)
        w2i = pkl.load(open(vocab_path, "rb"))
        _, test_data_loader, _, _ = load_data(
            sst_dir, batch_size, max(kernel_sizes), w2i)
        classifier = CNNClassifier(kernel_sizes=kernel_sizes, stride=stride,
                                   num_filters=num_filters, dropout_prob=dropout_prob, n_classes=n_classes,
                                   embedding_file=None, embedding_dim=embedding_dim, w2i=w2i)
        classifier.load_state_dict(classifier_parameters)
        classifier = classifier.eval()
        n_matches, total = 0, 0
        for batch in test_data_loader:
            sentence, true_labels = batch['sentence'], batch['label']
            predictions = classifier(sentence[:, :])
            predicted_labels = np.argmax(predictions.data.numpy(), axis=1)
            n_matches += np.sum(true_labels.data.numpy() == predicted_labels)
            total += true_labels.shape[0]
        print(f"Test Accuracy = {n_matches/total}")


if __name__ == "__main__":
    fire.Fire(Trainer)
