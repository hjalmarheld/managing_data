import pandas as pd
import numpy as np

import selenium.common.exceptions
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

import os


def scrape_insee(gecko_path: str = '/Users/charlesnicholas/Documents/Quinten project/managing_data/geckodriver') \
        -> pd.DataFrame:
    """This function scrapes the 'insee' website"""
    # DataFrame that will be filled with the scraped information
    text = pd.DataFrame(columns=['section_title', 'section_text',
                                 'division_title', 'division_text',
                                 'group_title', 'group_text',
                                 'class_title', 'class_text',
                                 'sub_class_title', 'sub_class_text'])

    # You need to provide your own path to your geckodriver here. If you do not have one installed yet, download it :)
    # We open a new Firefox instance
    driver = webdriver.Firefox(executable_path=gecko_path)
    # Loading the page
    driver.get('https://www.insee.fr/fr/metadonnees/nafr2/section/A?champRecherche=false')
    # We wait until the element we are looking for has been loaded
    WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, 'arbre-racine')))
    root = driver.find_element(By.CLASS_NAME, 'arbre-racine')
    sections = root.find_elements(By.CSS_SELECTOR, 'li')
    # We loop over the different NAF navigation levels
    for section in sections:
        section.click()
        content = driver.find_element(By.XPATH, '/html/body/main/section/div[2]/div[2]/div[2]/div/div/div[1]/div[1]')
        # Here, we store the text in the dictionary with the title as key and the text as value. It is important to note
        # that the text can consists out of several paragraphs which will have to be considered in the text processing
        try:
            s_list = [content.find_element(By.XPATH, './h2').text, content.find_element(By.XPATH, './div').text]
            # text[content.find_element(By.XPATH, './h2').text] = content.find_element(By.XPATH, './div').text
        except selenium.common.exceptions.NoSuchElementException:
            s_list = [content.find_element(By.XPATH, './h2').text, np.nan]
            # pass
        divisions = section.find_element(By.TAG_NAME, 'ul').find_elements(By.CSS_SELECTOR, 'li')
        for division in divisions:
            if not division.text:
                continue
            division.click()
            content = driver.find_element(By.XPATH, '/html/body/main/section/div[2]/div[2]/div[2]/div/div/div[1]/div[1]')
            try:
                d_list = [content.find_element(By.XPATH, './h2').text, content.find_element(By.XPATH, './div').text]
                # text[content.find_element(By.XPATH, './h2').text] = content.find_element(By.XPATH, './div').text
            except selenium.common.exceptions.NoSuchElementException:
                d_list = [content.find_element(By.XPATH, './h2').text, np.nan]
                # pass
            groups = division.find_element(By.TAG_NAME, 'ul').find_elements(By.CSS_SELECTOR, 'li')
            for group in groups:
                if not group.text:
                    continue
                group.click()
                content = driver.find_element(By.XPATH,
                                              '/html/body/main/section/div[2]/div[2]/div[2]/div/div/div[1]/div[1]')
                try:
                    g_list = [content.find_element(By.XPATH, './h2').text, content.find_element(By.XPATH, './div').text]
                    # text[content.find_element(By.XPATH, './h2').text] = content.find_element(By.XPATH, './div').text
                except selenium.common.exceptions.NoSuchElementException:
                    g_list = [content.find_element(By.XPATH, './h2').text, np.nan]
                    # pass
                classes = group.find_element(By.TAG_NAME, 'ul').find_elements(By.CSS_SELECTOR, 'li')
                for classe in classes:
                    if not classe.text:
                        continue
                    classe.click()
                    content = driver.find_element(By.XPATH,
                                                  '/html/body/main/section/div[2]/div[2]/div[2]/div/div/div[1]/div[1]')
                    try:
                        c_list = [content.find_element(By.XPATH, './h2').text, content.find_element(By.XPATH, './div').text]
                        # text[content.find_element(By.XPATH, './h2').text] = content.find_element(By.XPATH, './div').text
                    except selenium.common.exceptions.NoSuchElementException:
                        c_list = [content.find_element(By.XPATH, './h2').text, np.nan]
                        # pass
                    sub_classes = classe.find_element(By.TAG_NAME, 'ul').find_elements(By.CSS_SELECTOR, 'li')
                    for sub_class in sub_classes:
                        if not sub_class.text:
                            continue
                        sub_class.click()
                        content = driver.find_element(By.XPATH,
                                                      '/html/body/main/section/div[2]/div[2]/div[2]/div/div/div[1]/div[1]')
                        try:
                            sc_list = [content.find_element(By.XPATH, './h2').text,
                                       content.find_element(By.XPATH, './div').text]
                            text.loc[len(text)] = s_list + d_list + g_list + c_list + sc_list
                            # text[content.find_element(By.XPATH, './h2').text] = content.find_element(By.XPATH,
                            #                                                                          './div').text
                        except selenium.common.exceptions.NoSuchElementException:
                            try:
                                sc_list = [content.find_element(By.XPATH, './h2').text,
                                           driver.find_element(By.CLASS_NAME, 'comprend').text]
                                text.loc[len(text)] = s_list + d_list + g_list + c_list + sc_list
                                # text[content.find_element(By.XPATH, './h2').text] = driver.find_element(By.CLASS_NAME,
                                #                                                                         'comprend').text
                            except selenium.common.exceptions.NoSuchElementException:
                                sc_list = [content.find_element(By.XPATH, './h2').text,
                                           np.nan]
                                text.loc[len(text)] = s_list + d_list + g_list + c_list + sc_list
                                # pass
                    WebDriverWait(classe, 5).until(EC.presence_of_element_located((By.XPATH, './a')))
                    try:
                        classe.find_element(By.XPATH, './a').click()
                    except selenium.common.exceptions.ElementClickInterceptedException:
                        pass
                try:
                    group.find_element(By.XPATH, './a').click()
                except selenium.common.exceptions.ElementClickInterceptedException:
                    pass
            try:
                division.find_element(By.XPATH, './a').click()
            except selenium.common.exceptions.ElementClickInterceptedException:
                pass
        try:
            section.find_element(By.XPATH, './a').click()
        except selenium.common.exceptions.ElementClickInterceptedException:
            pass

    return text


def assign_codes(data: pd.DataFrame) -> pd.DataFrame:
    """Extracting the codes from the titles"""
    data = data.copy()
    data['section_code'] = data['section_title'].str.extract('\s(\w)\s')
    data['division_code'] = data['division_title'].str.extract('\s(\w\w)\s')
    data['group_code'] = data['group_title'].str.extract('\s(\w\w\.\w)\s')
    data['class_code'] = data['class_title'].str.extract('\s(\w\w\.\w\w)\s')
    data['sub_class_code'] = data['sub_class_title'].str.extract('\s(\w\w\.\w\w\w)\s')
    return data


file_name = 'insee_info_df.pkl'
if file_name in os.listdir():
    text = pd.read_pickle(file_name)
else:
    text = scrape_insee()
    text = assign_codes(text)
# Since the codes we get from the website are in the new NAF standard and the ones we received from Quinten are in the
# old standard we have to map them. Luckily there is a mapping available on the website -> we load it here
mapping = pd.read_excel('table_NAF2-NAF1.xls')[['NAF\nrév. 2', 'NAF\nrév. 1']]
# Some of the codes in the mapping end in a small "p", we remove it
mapping['NAF\nrév. 2'] = mapping['NAF\nrév. 2'].str.replace('p', '')
# The codes in the files we received from Quentin don't contain a dot in the code -> remove it from the old code as well
mapping['NAF\nrév. 1'] = mapping['NAF\nrév. 1'].str.replace('\.|p', '')
# Merging the mapping with the scraped data
df = pd.merge(text, mapping, left_on='sub_class_code', right_on='NAF\nrév. 2', how='outer')
# Since Quinten data is sometimes using the old and sometimes the new codes, but always without the dot, we remote it
# here from the new identifiers as well (we did it for the old one above already)
df['NAF\nrév. 2'] = df['NAF\nrév. 2'].str.replace('\.', '')

# Now, we can load the files from Quinten
naf_m = pd.read_csv('naf_mapping.csv', sep=';', encoding='latin-1')
naf_a = pd.read_csv('naf_activite.csv', sep='|')
# Since for now we can only map through the NAF code, we remove all records that do not contain said code. However,
# some records that do not have a NAF code do have a SIREN identifier, that could possibly mapped through an API or
# scraping to a NAF code
naf_a = naf_a[~naf_a['NAF_CODE'].isna()]
# One records has a NAF code that cannot be mapped -> we retrieve its actual NAF code online and replace its wrong value
naf_a.loc[naf_a['SIREN'] == 329328645, 'NAF_CODE'] = '4511Z'
# Since half of the identifiers is in the old NAF code and the other half in the new standard, we have to perform a one-
# sided merge with one first and then merge on the other.
df_new = pd.merge(naf_a.loc[naf_a['NAF_CODE'].apply(lambda x: len(x) == 5)], df[~df['NAF\nrév. 2'].duplicated()],
                  left_on='NAF_CODE', right_on='NAF\nrév. 2', how='left')
df_old = pd.merge(naf_a.loc[naf_a['NAF_CODE'].apply(lambda x: len(x) == 4)], df,
                  left_on='NAF_CODE', right_on='NAF\nrév. 1')
df = pd.concat([df_new, df_old])
print('Done')
