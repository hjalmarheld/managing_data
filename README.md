# NAFC: Classifying news into the nomenclature d'activités française

This repository provides a model to classify news into the french NAF codes at division level. It also provides an already labeled dataset to train the model on as well as helper function to scrape more data and easily transform it to the required format.

## Installation
Either clone this github repository or download all its files to the folder you want this project to be in.
Just like with any project you should first create a virtual environment to avoid messing up the pre-installed version. There are several different ways of doing this, one of which is using the venv module that comes pre-shipped with python. It will install the environment by default in your current working directory. Therefore, the first step is to open the terminal and navigating to the directory you want the environment to be in by pasting the following line in your terminal (and replacing the text between the double quotes with the path to your dedicated folder):
```sh
# MacOS and Windows
cd "path/to/the/desired/folder"
```
Now that we are in the desired directory we can create the environment by writing the following line in the terminal (and replacing "name_of_your_environment" with whatever you want to call your environment):
```sh
# MacOS
python3 -m venv name_of_your_environment
# Windows
python -m venv name_of_your_environment
```
Having created the environment, we can activate it with the following line in the terminal:
```sh
# MacOS
source name_of_your_environment/bin/activate
# Windows
name_of_your_environment/Scripts/activate.bat
```
This will prepend the environment name in parenthesis in the terminal, telling us that we are now in the virtual environment.
For more information on the venv module refer to the offical venv [**documentation**](https://docs.python.org/3/library/venv.html#module-venv).
To create virtual environments using conda, refer to their [**documentation**](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

Finally, we can install the necessary packages from the requirements.txt file that comes with this repository, the following line does just that:
```sh
# MacOS and Windows
pip install -r /path/to/requirements.txt
```

# Main Features
- A pytorch classification model built on top of the pretrained transformer CamemBERT with an additional dense layer
- Functions to train and evaluate the model
- Labeled data to train the model on
- Functions to scrape the description of the NAF codes
- Functions to scrape news from bing
- Functions to perform data augmentation such as synonym replacement or back-translation
- Functions to plot the results

# Examples
A Jupyter Notebook that showcases the use of the different classes and function has been created. It can be found int he "notebooks"  folder

# Background
This project was created for a week-long Kaggle challenge managed by [**Quinten**](https://www.quinten.ai/en/). It aimed at classifying news, tweeks or reports into 88 different NAF divisions which describe a company's main activities.