import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import Stemmer
import requests
import bs4 as bs
from tqdm import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger

import selenium.common.exceptions
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


def stem(x: str) -> str:
    """This function turns all words in the provided parameter x to their stem form."""
    x = x.lower()
    x = x.replace("l'", '').replace("d'", '')
    string = ' '.join(
        Stemmer.Stemmer('french').stemWords(
            [x for x in x.split(' ') if x not in stopwords.words('french')])
        )
    string = string.replace(',', '')
    return string


def search(search_term: str) -> pd.DataFrame:
    """This function has to be used in combination with the right global parameters, and then searches bing for news on
    the search_term value."""
    params["q"] = search_term
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    df = pd.json_normalize(response.json())
    df = pd.DataFrame(df['value'][0])
    return df


def clean_html(x: str) -> str:
    """This function extracts the text from an HTML string."""
    soup = bs.BeautifulSoup(x, 'html.parser')
    return soup.get_text()


def scrape_insee(gecko_path: str = './geckodriver') \
        -> pd.DataFrame:
    """This function scrapes the 'insee' website for the NAF code descriptions."""
    # DataFrame that will be filled with the scraped information
    text = pd.DataFrame(columns=['section_title', 'section_text',
                                 'division_title', 'division_text',
                                 'group_title', 'group_text',
                                 'class_title', 'class_text',
                                 'sub_class_title', 'sub_class_text'])

    # You need to provide your own path to your geckodriver here. If you do not have one installed yet, download it :)
    # We open a new Firefox instance
    driver = webdriver.Firefox(Service(executable_path=gecko_path))
    # Loading the page
    driver.get('https://www.insee.fr/fr/metadonnees/nafr2/section/A?champRecherche=false')
    # We wait until the element we are looking for has been loaded
    WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, 'arbre-racine')))
    root = driver.find_element(By.CLASS_NAME, 'arbre-racine')
    sections = root.find_elements(By.CSS_SELECTOR, 'li')
    description_xpath = '/html/body/main/section/div[2]/div[2]/div[2]/div/div/div[1]/div[1]'
    # We loop over the different NAF navigation levels (section -> division -> group -> class -> sub-class) and retrieve
    # for each element at each level the text description
    for section in sections:
        # We click on the section
        section.click()
        # We retrieve the HTML element that holds the title and text describing the seciton
        content = driver.find_element(By.XPATH, description_xpath)
        # Here, we store the text in the dictionary with the title as key and the text as value. It is important to note
        # that the text can consists out of several paragraphs which will have to be considered in the text processing
        try:
            s_list = [content.find_element(By.XPATH, './h2').text, content.find_element(By.XPATH, './div').text]
        except selenium.common.exceptions.NoSuchElementException:
            # If nothing could be found we just use np.nan values
            s_list = [content.find_element(By.XPATH, './h2').text, np.nan]
        # Here we get the navigation element of the next lower lever, in this case the division level
        divisions = section.find_element(By.TAG_NAME, 'ul').find_elements(By.CSS_SELECTOR, 'li')
        # Now, we loop over all divisions in the current section
        for division in divisions:
            if not division.text:
                continue
            division.click()
            content = driver.find_element(By.XPATH, description_xpath)
            try:
                d_list = [content.find_element(By.XPATH, './h2').text, content.find_element(By.XPATH, './div').text]
            except selenium.common.exceptions.NoSuchElementException:
                d_list = [content.find_element(By.XPATH, './h2').text, np.nan]
            groups = division.find_element(By.TAG_NAME, 'ul').find_elements(By.CSS_SELECTOR, 'li')
            for group in groups:
                if not group.text:
                    continue
                group.click()
                content = driver.find_element(By.XPATH, description_xpath)
                try:
                    g_list = [content.find_element(By.XPATH, './h2').text, content.find_element(By.XPATH, './div').text]
                except selenium.common.exceptions.NoSuchElementException:
                    g_list = [content.find_element(By.XPATH, './h2').text, np.nan]
                classes = group.find_element(By.TAG_NAME, 'ul').find_elements(By.CSS_SELECTOR, 'li')
                for classe in classes:
                    if not classe.text:
                        continue
                    classe.click()
                    content = driver.find_element(By.XPATH, description_xpath)
                    try:
                        c_list = [content.find_element(By.XPATH, './h2').text, content.find_element(By.XPATH, './div').text]
                    except selenium.common.exceptions.NoSuchElementException:
                        c_list = [content.find_element(By.XPATH, './h2').text, np.nan]
                    sub_classes = classe.find_element(By.TAG_NAME, 'ul').find_elements(By.CSS_SELECTOR, 'li')
                    for sub_class in sub_classes:
                        if not sub_class.text:
                            continue
                        sub_class.click()
                        content = driver.find_element(By.XPATH, description_xpath)
                        try:
                            sc_list = [content.find_element(By.XPATH, './h2').text,
                                       content.find_element(By.XPATH, './div').text]
                            text.loc[len(text)] = s_list + d_list + g_list + c_list + sc_list
                        except selenium.common.exceptions.NoSuchElementException:
                            try:
                                sc_list = [content.find_element(By.XPATH, './h2').text,
                                           driver.find_element(By.CLASS_NAME, 'comprend').text]
                                text.loc[len(text)] = s_list + d_list + g_list + c_list + sc_list
                            except selenium.common.exceptions.NoSuchElementException:
                                sc_list = [content.find_element(By.XPATH, './h2').text,
                                           np.nan]
                                text.loc[len(text)] = s_list + d_list + g_list + c_list + sc_list
                    WebDriverWait(classe, 5).until(EC.presence_of_element_located((By.XPATH, './a')))
                    # Here we move back up one level, in this case from the sub-class navigation to the class navigation
                    # by clicking on the current class
                    classe.find_element(By.XPATH, './a').click()
                # Here we move from class to group level
                group.find_element(By.XPATH, './a').click()
            division.find_element(By.XPATH, './a').click()
        section.find_element(By.XPATH, './a').click()

    return text


def assign_codes(data: pd.DataFrame) -> pd.DataFrame:
    """This function extracts the codes from the titles in the DataFrame containing the scraped NAF descriptions."""
    data = data.copy()
    data['section_code'] = data['section_title'].str.extract('\s(\w)\s')
    data['division_code'] = data['division_title'].str.extract('\s(\w\w)\s')
    data['group_code'] = data['group_title'].str.extract('\s(\w\w\.\w)\s')
    data['class_code'] = data['class_title'].str.extract('\s(\w\w\.\w\w)\s')
    data['sub_class_code'] = data['sub_class_title'].str.extract('\s(\w\w\.\w\w\w)\s')
    return data


def combineLabels(df: pd.DataFrame) -> pd.DataFrame:
    """"
    masking a random word in a string
    args:
        - text_line: string
    """

    df['description'] = df[['naf5_label', 'naf4_label', 'naf3_label', 'naf2_label']].apply(
        lambda x: ' '.join(x.dropna().astype(str)),
        axis=1)
    df['description'] = df['description'].str.lower()
    return (df)


class Augment:
    def __init__(self):
        pass

    def maskRandomWord(self, text_line: str) -> str:
        """
        masking a random word in a string
        args:
            - text_line: string
        """

        text_array = text_line.split()
        if len(text_array) == 1:
            masked_token_index = 0
        else:
            masked_token_index = np.random.randint(0, len(text_array) - 1)
        text_array[masked_token_index] = '<mask>'
        output_text = ' '.join(text_array)
        return output_text

    def generateSyntheticTexts(self, model_mask, text: str, number_of_examples: int = 5) -> list:
        """
        using a transformer model to replace a masked word
        args:
            - text: masked string
            - number_of_examples: # replacement words to take from model
            - model_mask: transformer model pipeline to use
        """

        output = []
        unmaskedText = model_mask(text)
        if number_of_examples >= len(unmaskedText):
            number_of_examples = len(unmaskedText)
        for i in range(number_of_examples):
            unmasked = unmaskedText[i]['sequence']
            output.append(unmasked)
        return output

    def nerExtract(self, df: pd.DataFrame, tagger: SequenceTagger) -> pd.DataFrame:
        """"
        extracting key words from descriptor phrase
        args:
            - df: input dataframe
            - tagger: model tagger. Have to preload in some model that can extract NERs
        """

        df['NER'] = ''
        for row in tqdm(df.itertuples(), total=df.shape[0], position=0, leave=True):
            sentence = Sentence(row.synthText)
            tagger.predict(sentence)
            # iterate over entities and output
            entities = []
            for entity in sentence.get_spans('ner'):
                entities.append(entity.text)
            df.at[row.Index, 'NER'] = ' '.join(entities)
        return (df)