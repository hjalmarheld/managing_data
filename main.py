from cProfile import label
from matplotlib.backend_bases import key_press_handler
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, CamembertModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import os
from function_camembert_model import *
import ipdb
from config import *

general_path = os.path.dirname(os.getcwd())
name_file = "labelled articles cleaned.csv"
path_data = os.path.join(os.getcwd(),"data","external", name_file)
path_mapping = os.path.join(os.getcwd(),"data","raw", "naf_mapping.csv")
path_predictions = os.path.join(os.getcwd(),"data","test.csv")
EPOCHS = 2
LR = 1e-5
save = True
increase_data = False
batch_size=4

if __name__ == "__main__":
    df = pd.read_csv(path_data, delimiter=",").dropna().drop(columns=["Unnamed: 0"])
    if name_file == "naf_activite.csv":
        df_mapping = pd.read_csv(path_mapping, delimiter=";", encoding="latin-1")
        df_mapping.rename(columns={"naf5": "NAF_CODE"}, inplace=True)
        df_merged = df.merge(df_mapping, on="NAF_CODE", how="inner")
    
    else:
        df_merged = df.copy()
    if increase_data:
        df_merged_bis = df_merged.copy()
        df_merged_bis.columns = ["company","text","title","naf"]
        df_merged = pd.concat([df_merged, df_merged_bis[df_merged.columns]])
        
    idxes = np.random.choice(np.arange(len(df_merged)), 400)
    df_merged = df_merged.iloc[idxes, :]
    model = "camembert-base"
    tokenizer = AutoTokenizer.from_pretrained(model)

    labels = {
        label: i
        for i, label in enumerate(df_merged["naf"].sort_values().unique())
    }
    labels_list = [key for key in labels.keys()]

    np.random.seed(112)
    df_train, df_val, df_test = np.split(
        df_merged.sample(frac=1, random_state=42),
        [int(0.8 * len(df_merged)), int(0.9 * len(df_merged))],
    )
    del df_merged
    model = CamemBertClassifierShort(number_labels=len(labels_list))
    train_data = Dataset(df_train, tokenizer, labels=labels)
    
    train(model, df_train, df_val, LR, EPOCHS, tokenizer, labels,use_samplers=True,batch_size=batch_size)
    acc_test, predictions = evaluate(model, df_test,tokenizer, labels=labels)
    df_test_final = pd.read_csv(path_predictions)
    df_test_final["naf"] = df_train["naf"].iloc[0]

    predictions = predict(model,df_test_final, tokenizer, labels, batch_size)
    values = []
    for i in range(len(predictions)):
        for j in range(predictions[i].shape[0]):
            values.append(list(predictions[i][j]))
    df_predictions = pd.DataFrame(values)
    df_predictions.columns = [f"naf_code = {naf_code}" for naf_code in labels.keys()]

    if save:
        try:
            df_predictions.to_csv('predictions.csv',index=False)
        except:
            ipdb.set_trace()
    ipdb.set_trace()
