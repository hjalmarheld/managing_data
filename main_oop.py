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
# from config import *
import config
from model_creator import final_classifier

general_path = os.path.dirname(os.getcwd())
path_data = os.path.join(os.getcwd(), "data", "external", config.name_file)
path_mapping = os.path.join(os.getcwd(), "data", "raw", "naf_mapping.csv")
path_predictions = os.path.join(os.getcwd(), "data", "test.csv")

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
        df_merged_bis.columns = ["company", "text", "title", "naf"]
        df_merged = pd.concat([df_merged, df_merged_bis[df_merged.columns]])

    idxes = np.random.choice(np.arange(len(df_merged)), size_training)
    df_merged = df_merged.iloc[idxes, :]
    model = "camembert-base"
    tokenizer = AutoTokenizer.from_pretrained(model)

    labels = {
        label: i for i, label in enumerate(df_merged["naf"].sort_values().unique())
    }
    labels_list = [key for key in labels.keys()]

    np.random.seed(112)
    df_train, df_val, df_test = np.split(
        df_merged.sample(frac=1, random_state=42),
        [int(0.8 * len(df_merged)), int(0.9 * len(df_merged))],
    )
    del df_merged
    dl_model = CamemBertClassifierShort(number_labels=len(labels_list))
    model = final_classifier(dl_model, tokenizer, labels)

    model.train(
        df_train,
        df_val,
        LR,
        EPOCHS,
        use_samplers=config.use_samplers,
        batch_size=config.batch_size,
        text_column=config.text_column
    )
    acc_test, predictions = model.evaluate(df_test,batch_size=config.batch_size,column_labels=config.column_labels, text_column=config.text_column)
    df_test_final = pd.read_csv(path_predictions)
    df_test_final["naf"] = df_train["naf"].iloc[0]

    predictions = model.predict_proba(
        df_test_final, config.batch_size,test_column = config.test_column
    )
    values = []
    for i in range(len(predictions)):
        for j in range(predictions[i].shape[0]):
            values.append(list(predictions[i][j]))
    df_predictions = pd.DataFrame(values)
    df_predictions.columns = [f"naf_code = {naf_code}" for naf_code in labels.keys()]

    if save:
        try:
            df_predictions.to_csv(name_saving, index=False)
        except:
            ipdb.set_trace()
    ipdb.set_trace()
