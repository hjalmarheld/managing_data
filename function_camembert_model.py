import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel,CamembertModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import os
import ipdb
import sys

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df,tokenizer,labels):

        self.labels = [labels[label] for label in df["naf2_label"].values]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['ACTIVITE']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class CamemBertClassifier(nn.Module):

    def __init__(self, number_labels,dropout=0.5,model = "camembert-base",):

        super(CamemBertClassifier, self).__init__()

        self.bert = CamembertModel.from_pretrained(model)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, number_labels)
        self.relu = nn.ReLU()
        # self.softmax_final_layer = nn.Softmax(dim=1)

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        # final_layer = self.softmax_final_layer(final_layer)

        return final_layer

def train(model, train_data, val_data, learning_rate, epochs,tokenizer,labels):

    train, val = Dataset(train_data,tokenizer,labels), Dataset(val_data,tokenizer,labels)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=8)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)
    # ipdb.set_trace()
    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0
            try:
                for train_input, train_label in tqdm(train_dataloader):
                    train_label = train_label.to(device)
                    mask = train_input['attention_mask'].to(device)
                    input_id = train_input['input_ids'].squeeze(1).to(device)
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
            except:
                print("error")
                # ipdb.set_trace()
                sys.exit()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_acc_train / len(train_data): .3f} | Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {total_acc_val / len(val_data): .3f}')

def evaluate(model, test_data):

    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              acc = (output.argmax(dim=1) == test_label).sum().item()
              
              total_acc_test += acc
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')   

if __name__=="__main":
    datapath = os.path.join("naf-prediction","examples.csv")
    df = pd.read_csv(datapath).dropna()
    # df.head()
