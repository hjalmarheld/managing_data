import pandas as pd
from sklearn.utils import shuffle
import torch
import numpy as np
from traitlets import Bool
from transformers import BertTokenizer, BertModel, CamembertModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch import nn
from torch.optim import Adam
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
import os
import ipdb
import sys


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, labels: dict,colonne_labels:str="naf"):
        """
        Creation of a dataset loader with batches compatible with Pytorch.
        args:
            - df : dataframe containing texts and labels
            - tokenizer : huggingface tokenizer
            - labels : dict containing for each label a numerical encoding (similar to OrdinalEncoding)
            - colonne_labels : name of columns containing the labels
        """

        self.labels = [labels[label] for label in df[colonne_labels].values]
        self.labels_dict = labels
        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            for text in df["text"]
        ]

    def classes(self):
        """
        returns the labels of elements of the Dataset.
        """
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        """
        Fetch a batch of labels
        args :
            - idx : set of indexes to query
        """
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        """
        Fetch a batch of inputs
        args :
            - idx : set of indexes to query
        """
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
    
    def classes_imbalance_sampler(self):
        targets = self.labels
        class_sample_count = np.array(
        [len(np.where(targets == t)[0]) for t in np.arange(0,max(targets)+1)])
        weight = 1. / (class_sample_count + 0.1)
        # ipdb.set_trace()
        weights = list()
        for t in targets:
            try : 
                weights.append(weight[t])
            except:
                ipdb.set_trace()
        samples_weight = np.array(weights)
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return sampler
    
class CamemBertClassifierShort(nn.Module):
    def __init__(
        self,
        number_labels: int,
        dropout: float = 0.5,
        model: str = "camembert-base",
    ):
        """
        Class generating a DL model upon the CamembertModel class from hugging face
        args:
            - int : number of categories to predict
            - dropout : float in [0,1), dropout rate between layers to avoid over-fitting
            - model : str, name of HuggingFace model to use
        """
        super(CamemBertClassifierShort, self).__init__()

        self.bert = CamembertModel.from_pretrained(model)
        self.dropout = nn.Dropout(dropout)
        self.linear_short = nn.Linear(768, number_labels)
        self.relu = nn.ReLU()
        self.dropout_second = nn.Dropout(dropout)
        

        # self.softmax_final_layer = nn.Softmax(dim=1)

    def forward(self, input_id: torch.tensor, mask: torch.tensor):
        """
        compute the output of forward run from model given input_id and mask
        args:
            - input_id : numerical embedding of words in sentenced, output of tokenizer
            - mask : whether ids are mask or not.

        returns:
            - final_layer : final_layer of the model
        """
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear_short(dropout_output)
        final_layer = self.relu(linear_output)

        # final_layer = self.softmax_final_layer(final_layer)

        return final_layer


class CamemBertClassifier(nn.Module):
    def __init__(
        self,
        number_labels: int,
        dropout: float = 0.5,
        model: str = "camembert-base",
    ):
        """
        Class generating a DL model upon the CamembertModel class from hugging face
        args:
            - int : number of categories to predict
            - dropout : float in [0,1), dropout rate between layers to avoid over-fitting
            - model : str, name of HuggingFace model to use
        """
        super(CamemBertClassifier, self).__init__()

        self.bert = CamembertModel.from_pretrained(model)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 250)
        self.relu = nn.ReLU()
        self.dropout_second = nn.Dropout(dropout)
        self.second_linear = nn.Linear(250, number_labels)
        self.relu_2 = nn.ReLU()
        self.softmax_layer = nn.Softmax(dim=1)
        self.linear_short == nn.Linear(768, number_labels)

        # self.softmax_final_layer = nn.Softmax(dim=1)

    def forward(self, input_id: torch.tensor, mask: torch.tensor):
        """
        compute the output of forward run from model given input_id and mask
        args:
            - input_id : numerical embedding of words in sentenced, output of tokenizer
            - mask : whether ids are mask or not.

        returns:
            - final_layer : final_layer of the model
        """
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        relu_layer = self.relu(linear_output)
        # second_dropout = self.dropout_second(relu_layer)
        second_linear_output = self.second_linear(relu_layer)
        final_layer = self.relu_2(second_linear_output)

        # final_layer = self.softmax_final_layer(final_layer)

        return final_layer
    
    def train(
        self,
        train_data: Dataset,
        val_data: Dataset,
        learning_rate: float,
        epochs: int,
        tokenizer,
        labels: dict,
        use_samplers:Bool = False
    ):
        pass
    
    def predict(self):
        pass
    
    def evaluate(self):
        pass


def train(
    model: CamemBertClassifier,
    train_data: Dataset,
    val_data: Dataset,
    learning_rate: float,
    epochs: int,
    tokenizer,
    labels: dict,
    use_samplers:Bool = False
):
    """
    Training of model defined with CamemBertClassifier class.
    args:
        - train_data: dataframe containing training data
        - val_data: dataframe containing validation data
        - learning_rate : float>0., learning_rate used in backpropagation algorithm
        - epochs : int, number of epochs to train the model
        - tokenizer : HuggingFace tokenizer used in Dataset class
        - labels : dictionnary containing the mapping for each class
    """
    train, val = Dataset(train_data, tokenizer, labels), Dataset(
        val_data, tokenizer, labels
    )

    if use_samplers:
        train_sampler = train.classes_imbalance_sampler()
        val_sampler = val.classes_imbalance_sampler()
        train_dataloader = torch.utils.data.DataLoader(train, batch_size=64,sampler=train_sampler)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=64, sampler=val_sampler)
    else:
        train_dataloader = torch.utils.data.DataLoader(train, batch_size=32,shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=32, shuffle=True)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # ipdb.set_trace()
    if use_cuda:

        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input["attention_mask"].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)
            # ipdb.set_trace()
            output = model(input_id, mask)
            # print(output)
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                mask = val_input["attention_mask"].to(device)
                input_id = val_input["input_ids"].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_acc_train / len(train_data): .3f} | Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {total_acc_val / len(val_data): .3f}"
        )


def evaluate(model: CamemBertClassifier, test_data: pd.DataFrame, tokenizer, labels):
    """
    Evaluates performance of CamembertClassifier trained with train function

    args:
        - model : DL model trained previously
        - test_data : dataframe containing test data

    returns:
        - total_acc_test : float, accuracy of the model on the test set

    """
    test = Dataset(test_data,tokenizer, labels)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    predictions = []
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

            test_label = test_label.to(device)
            mask = test_input["attention_mask"].to(device)
            input_id = test_input["input_ids"].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            predictions.append(nn.functional.softmax(output,dim=1).detach().cpu().numpy())
            total_acc_test += acc

    print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")
    return total_acc_test, predictions

def predict(model: CamemBertClassifier, test_data: pd.DataFrame,tokenizer, labels):
    """
    Evaluates performance of CamembertClassifier trained with train function

    args:
        - model : DL model trained previously
        - test_data : dataframe containing test data

    returns:
        - total_acc_test : float, accuracy of the model on the test set

    """
    test = Dataset(test_data,tokenizer,labels)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=32)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    predictions = []
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

            test_label = test_label.to(device)
            mask = test_input["attention_mask"].to(device)
            input_id = test_input["input_ids"].squeeze(1).to(device)

            output = model(input_id, mask)
            output = nn.functional.softmax(output, dim=1)
            predictions.append(output.detach().cpu().numpy())

    return predictions