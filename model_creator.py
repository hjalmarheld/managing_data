from lib2to3.pgen2 import token
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
from function_camembert_model import *
import config

class classifier:
    
    def __init__(self,model:CamembertCLassifier,tokenizer,labels:dict) -> None:
        """Instance of classifier class. Instead of relying on functions, we created this class. the class is used to train the model and make predictions for a given dataset once on the model has been trained
        args:
            - model : Pytorch model built upon Camembert tokenizer
            - tokenizer : Camember tokenizer from HuggingFace
            - labels : dict containing the mapping for the labels
        """
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels
        self.is_model_trained = False


    def train(
        self,
        train_data: Dataset,
        val_data: Dataset,
        learning_rate: float,
        epochs: int,
        use_samplers:Bool = False,
        batch_size:int = config.batch_size,
        text_column:str = config.text_column
    ):
        """
        method to launch training of the model
        """
        train, val = Dataset(train_data, self.tokenizer, self.labels, config.columns_labels,config.text_column), Dataset(
                val_data, self.tokenizer, self.labels, config.columns_labels,config.text_column
            )

        if use_samplers:
            train_sampler = train.classes_imbalance_sampler()
            val_sampler = val.classes_imbalance_sampler()
            train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size,sampler=train_sampler)
            val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size, sampler=val_sampler)
        else:
            train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size,shuffle=True)
            val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        criterion = nn.CrossEntropyLoss()
        # ipdb.set_trace()
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        # ipdb.set_trace()
        if use_cuda:

            self.model = self.model.cuda()
            criterion = criterion.cuda()

        for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0
            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.to(device)
                mask = train_input["attention_mask"].to(device)
                input_id = train_input["input_ids"].squeeze(1).to(device)
                # ipdb.set_trace()
                output = self.model(input_id, mask)
                # print(output)
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                self.model.zero_grad()
                batch_loss.backward()
                optimizer.step()

            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input["attention_mask"].to(device)
                    input_id = val_input["input_ids"].squeeze(1).to(device)

                    output = self.model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()

                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc

            print(
                f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_acc_train / len(train_data): .3f} | Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {total_acc_val / len(val_data): .3f}"
            )
            self.is_model_trained=True

    
    def predict_proba(self,test_data:pd.DataFrame,batch_size:int=config.batch_size,
                     column_labels:str=config.column_labels,
                     test_column:str=confif.test_column):
        """
        for a given dataset, returns the probabilities of belonging to the different classes
        
        args :
            - test_data : dataframe containing test data
            - batch_size: int > 0, batch_size for inputs forward in the model
            - column_labels: string, colonne of dataframe with labels, if the dataframe has no label, randomly create one (they will not be used for the predictions)
            - test_column: string
            
        returns : 
            - predictions : list containing for the probabilities for each input
        """
        if self.is_model_trained==False:
            raise AttributeError("the model has not yet been trained, train it first before predictions")
            sys.exit()
            
        test = Dataset(test_data,self.tokenizer,self.labels,column_labels, test_column)
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:
            self.model = self.model.cuda()

        predictions = []
        with torch.no_grad():

            for test_input, test_label in test_dataloader:

                test_label = test_label.to(device)
                mask = test_input["attention_mask"].to(device)
                input_id = test_input["input_ids"].squeeze(1).to(device)

                output = self.model(input_id, mask)
                predictions.append(output.detach().cpu().numpy())

        return predictions
    
    def predict(self,test_data):
        test = Dataset(test_data,self.tokenizer,self.labels)
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:
            self.model = self.model.cuda()

        predictions = []
        with torch.no_grad():

            for test_input, test_label in test_dataloader:

                test_label = test_label.to(device)
                mask = test_input["attention_mask"].to(device)
                input_id = test_input["input_ids"].squeeze(1).to(device)

                output = self.model(input_id, mask)
                predictions.append(output.argmax(dim=1).detach().cpu().numpy())

        return predictions
    
    def evaluate(self,test_data, column_labels,text_column):
        test = Dataset(test_data, self.tokenizer, self.labels,column_labels,text_column)

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
                predictions.append(
                    nn.functional.softmax(output, dim=1).detach().cpu().numpy()
                )
                total_acc_test += acc

        print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")
        return total_acc_test, predictions