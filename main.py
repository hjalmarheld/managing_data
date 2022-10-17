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


general_path = os.path.dirname(os.getcwd())
path_data = os.path.join(general_path,"naf_activite.csv")
path_mapping = os.path.join(general_path, "naf_mapping.csv")
# path_mapping = os.path.join()
if __name__=="__main__":
    # datapath = os.path.join(path_data)
    df = pd.read_csv(path_data,delimiter="|").dropna().drop(columns=["Unnamed: 0"])
    df_mapping = pd.read_csv(path_mapping,delimiter=';',encoding='latin-1')
    # df = df.drop(columns=["Unnamed: 0"])
    df_mapping.rename(columns={"naf5":"NAF_CODE"},inplace=True) 
    df_merged = df.merge(df_mapping,on="NAF_CODE",how="inner")
    idxes = np.random.choice(np.arange(len(df_merged)),30_000)
    df_merged = df_merged.iloc[idxes,:]
    model = "camembert-base"
    tokenizer = AutoTokenizer.from_pretrained(model)

    labels = {label : i for i,label in enumerate(df_merged["naf2_label"].sort_values().unique())}
    labels_list = [key for key in labels.keys()]
    np.random.seed(112)
    df_train, df_val, df_test = np.split(df_merged.sample(frac=1, random_state=42), 
                                        [int(.8*len(df_merged)), int(.9*len(df_merged))])
    del(df_merged)
    print(len(df_train),len(df_val), len(df_test))
    EPOCHS = 1
    model = CamemBertClassifier(number_labels=len(labels_list))
    LR = 1e-6
    train_data = Dataset(df_train,tokenizer,labels=labels)
    # ipdb.set_trace()           
    train(model, df_train, df_val, LR, EPOCHS,tokenizer,labels)