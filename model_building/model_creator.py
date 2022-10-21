import sys
import config
import torch
import pandas as pd
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from model_building.function_camembert_model import *


class final_classifier:
    def __init__(self, model: CamemBertClassifier, tokenizer, labels: dict) -> None:
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
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        learning_rate: float = config.LR,
        epochs: int = config.EPOCHS,
        use_samplers: bool = config.use_samplers,
        batch_size: int = config.batch_size,
        text_column: str = config.text_column,
    ):
        """
        method to launch training of the model.
        args:
            - train_data : pandas dataframe containing training data
            - val_data : pandas dataframe containing validation dataset
            - learning_rate : float > 0, learning rate
            - epochs : int, number of epochs for training
            - use_samplers : boolean, activate a class imbalance_sampler
            - batch_size : int, number of elements per batch
            - text_column : str, columns of dataframing containing the text

        returns:
            - nothing, however the is_model_trained attributed is changed to true
        """
        train, val = Dataset(
            train_data, self.tokenizer, self.labels, config.column_labels, text_column
        ), Dataset(
            val_data, self.tokenizer, self.labels, config.column_labels, text_column
        )

        if use_samplers:
            train_sampler = train.classes_imbalance_sampler()
            val_sampler = val.classes_imbalance_sampler()
            train_dataloader = torch.utils.data.DataLoader(
                train, batch_size=batch_size, sampler=train_sampler
            )
            val_dataloader = torch.utils.data.DataLoader(
                val, batch_size=batch_size, sampler=val_sampler
            )
        else:
            train_dataloader = torch.utils.data.DataLoader(
                train, batch_size=batch_size, shuffle=True
            )
            val_dataloader = torch.utils.data.DataLoader(
                val, batch_size=batch_size, shuffle=True
            )
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
            self.is_model_trained = True

    def predict_proba(
        self,
        test_data: pd.DataFrame,
        batch_size: int = config.batch_size,
        column_labels: str = config.column_labels,
        test_column: str = config.test_column,
    ):
        """
        for a given dataset, returns the probabilities of belonging to the different classes.
        Warnings :
            - if this method is used prior to the train one, the model will fail
            - to use this method with dataset without labels (i.e for example the final test of the challenge), one just needs to create a column with the config.test_column name and fill it with random values. These values will not be used.

        args :
            - test_data : dataframe containing test data
            - batch_size: int > 0, batch_size for inputs forward in the model
            - column_labels: string, colonne of dataframe with labels, if the dataframe has no label, randomly create one (they will not be used for the predictions)
            - test_column: string

        returns :
            - predictions : list containing the probabilities for each input
        """
        if self.is_model_trained == False:
            raise AttributeError(
                "the model has not yet been trained, train it first before predictions"
            )
            sys.exit()

        test = Dataset(
            test_data, self.tokenizer, self.labels, column_labels, test_column
        )
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

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

    def predict(
        self,
        test_data: pd.DataFrame,
        batch_size: int = config.batch_size,
        column_labels: str = config.column_labels,
        test_column: str = config.test_column,
    ):
        """
        for a given dataset, returns the predict class for each input.

        Warnings :
            - if this method is used prior to the train one, the model will fail
            - to use this method with dataset without labels (i.e for example the final test of the challenge), one just needs to create a column with the config.test_column name and fill it with random values. These values will not be used.


        args :
            - test_data : dataframe containing test data
            - batch_size: int > 0, batch_size for inputs forward in the model
            - column_labels: string, colonne of dataframe with labels, if the dataframe has no label, randomly create one (they will not be used for the predictions)
            - test_column: string

        returns :
            - predictions : list containing  the predicted class for each input
        """
        if self.is_model_trained == False:
            raise AttributeError(
                "the model has not yet been trained, train it first before predictions"
            )
            sys.exit()

        test = Dataset(
            test_data, self.tokenizer, self.labels, column_labels, test_column
        )
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

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

    def evaluate(
        self,
        test_data: pd.DataFrame,
        batch_size: int = config.batch_size,
        column_labels: str = config.column_labels,
        text_column: str = config.text_column,
    ):
        """
        for a given dataset, returns the total accuracy and the predicted class for each input.
        Warning : if this method is used prior to the train one, the model will fai

        args :
            - test_data : dataframe containing test data
            - batch_size: int > 0, batch_size for inputs forward in the model
            - column_labels: string, colonne of dataframe with labels, if the dataframe has no label, randomly create one (they will not be used for the predictions)
            - test_column: string

        returns :
            - predictions : list containing  the predicted class for each input
        """
        if self.is_model_trained == False:
            raise AttributeError(
                "the model has not yet been trained, train it first before predictions"
            )
            sys.exit()

        test = Dataset(
            test_data, self.tokenizer, self.labels, column_labels, text_column
        )

        test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:

            self.model = self.model.cuda()

        total_acc_test = 0
        predictions = []
        with torch.no_grad():

            for test_input, test_label in test_dataloader:

                test_label = test_label.to(device)
                mask = test_input["attention_mask"].to(device)
                input_id = test_input["input_ids"].squeeze(1).to(device)

                output = self.model(input_id, mask)

                acc = (output.argmax(dim=1) == test_label).sum().item()
                predictions.append(
                    nn.functional.softmax(output, dim=1).detach().cpu().numpy()
                )
                total_acc_test += acc

        print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")
        return total_acc_test, predictions
