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
from function_camembert_model import *
import ipdb

if __name__=="__main__":
    datapath = os.path.join("naf-prediction","examples.csv")
    df = pd.read_csv(datapath).dropna()
    model = "camembert-base"
    tokenizer = AutoTokenizer.from_pretrained(model)
    labels = {label : i for i,label in enumerate(df["naf2_label"].sort_values().unique())}
    labels_list = [key for key in labels.keys()]
    np.random.seed(112)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), 
                                        [int(.8*len(df)), int(.9*len(df))])

    print(len(df_train),len(df_val), len(df_test))
    EPOCHS = 5
    model = CamemBertClassifier(number_labels=len(labels_list))
    LR = 1e-6
    train_data = Dataset(df_train,tokenizer,labels=labels)
    # ipdb.set_trace()           
    train(model, df_train, df_val, LR, EPOCHS,tokenizer,labels)