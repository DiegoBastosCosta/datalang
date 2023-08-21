# Import necessary libraries
import base64
import spacy
import pandas as pd
from enum import Enum
from typing import List

from homepagev2 import HomePageProcessor

# Enum to represent pre-processing steps
class PreProcessingStepsV2(Enum):
    LOWERCASING = 'lowercasing'
    NON_ALPHA_REMOVAL = 'non alpha removal'
    NON_ASCII_REMOVAL = 'non ascii removal'
    PUNCTUATION_REMOVAL = 'punctuation removal'
    DIGITS_REMOVAL = 'digits removal'
    STOP_WORDS_REMOVAL = 'stop words removal'
    EMAIL_REMOVAL = 'email removal'
    URL_REMOVAL = 'url removal'
    LEMMATIZATION = 'lemmatization'

class SpacyModelNames(Enum):
    EN_CORE_WEB_SM = 'en_core_web_sm'
    PT_CORE_NEWS_SM = 'pt_core_news_sm'

# Preprocessing class
class PreProcessingV2:
    def __init__(self):
        self.steps = []

    def _load_spacy_model(self, model_name) -> spacy.Language:
        try:
            return spacy.load(model_name)
        except OSError:
            print(f"The model {model_name} is not installed. Attempting to download...")
            spacy.cli.download(model_name)
            print(f"Model {model_name} downloaded successfully.")
            return spacy.load(model_name)
        
    def set_model_name(self, model_name: SpacyModelNames):
        if model_name is None:
            return
        self.nlp = self._load_spacy_model(model_name)

    def add_step(self, step: PreProcessingStepsV2):
        self.steps.append(step)

    def lowercase(self, doc):
        processed = ' '.join([token.lower_ for token in doc])
        return processed

    def remove_non_alpha(self, doc):
        processed = ' '.join([token.text for token in doc if token.is_alpha])
        return processed

    def remove_non_ascii(self, doc):
        processed = ' '.join([token.text for token in doc if token.is_ascii])
        return processed

    def remove_punctuation(self, doc):
        # for token in doc:
        #     print("token.text=[{}]-token.is_punct=[{}]".format(token.text, token.is_punct))
        processed = ' '.join([token.text for token in doc if not token.is_punct])
        return processed

    def remove_digits(self, doc):
        processed = ' '.join([token.text for token in doc if not token.is_digit])
        return processed
    
    def remove_like_email(self, doc):
        processed = ' '.join([token.text for token in doc if not token.like_email])
        return processed
    
    def remove_like_url(self, doc):
        processed = ' '.join([token.text for token in doc if not token.like_url])
        return processed

    def remove_stop_words(self, doc):
        processed = ' '.join([token.text for token in doc if not token.is_stop])
        return processed

    def lemmatization(self, doc):
        processed = ' '.join([token.lemma_ for token in doc])
        return processed
    
    def preprocess(self, doc):
        for step in self.steps:
            if step == PreProcessingStepsV2.LOWERCASING:
                text = self.lowercase(doc)
            elif step == PreProcessingStepsV2.NON_ALPHA_REMOVAL:
                text = self.remove_non_alpha(doc)
            elif step == PreProcessingStepsV2.NON_ASCII_REMOVAL:
                text = self.remove_non_ascii(doc)
            elif step == PreProcessingStepsV2.PUNCTUATION_REMOVAL:
                text = self.remove_punctuation(doc)
            elif step == PreProcessingStepsV2.DIGITS_REMOVAL:
                text = self.remove_digits(doc)
            elif step == PreProcessingStepsV2.STOP_WORDS_REMOVAL:
                text = self.remove_stop_words(doc)
            elif step == PreProcessingStepsV2.EMAIL_REMOVAL:
                text = self.remove_like_email(doc)
            elif step == PreProcessingStepsV2.URL_REMOVAL:
                text = self.remove_like_url(doc)
            elif step == PreProcessingStepsV2.LEMMATIZATION:
                text = self.lemmatization(doc)

            # Update doc for the next step, only if needed
            if len(self.steps) > 1:
                doc = self.nlp(text)
        return text

    def preprocess_df(self, df: pd.DataFrame, columns: List[str], include_token_count: bool) -> pd.DataFrame:
        for col in columns:
            # Convert non-string data to string
            df[col] = df[col].astype(str)

            texts = df[col].tolist()
            docs = list(self.nlp.pipe(texts))

            processed_texts = [self.preprocess(doc) for doc in docs]
            df[col + '_processed'] = processed_texts

            if include_token_count:
                # Count tokens directly from docs before pre processing
                df[col + '_token_count'] = [len(doc) for doc in docs]
                # Count tokens directly from new docs after pre processing
                
                processed_texts = df[col+'_processed'].tolist()
                processed_docs = list(self.nlp.pipe(processed_texts))
                df[col + '_processed' + '_token_count'] = [len(doc) for doc in processed_docs]

        return df


import streamlit as st
import time
from st_pages import add_page_title

class PreProcessingUI:
    def __init__(self, preprocessor: PreProcessingV2, homepage_processor: HomePageProcessor):
        self.preprocessor = preprocessor
        self.homepage_processor = homepage_processor

    def select_columns(self, df):
        string_columns = [column for column in df.columns if df[column].dtype == 'object']
        return st.sidebar.multiselect("Select the text columns to process", options=string_columns, default=string_columns[0] if len(string_columns) > 0 else None)

    def select_preprocessing_steps(self):
        steps = [e.value for e in PreProcessingStepsV2]
        selected_steps = st.sidebar.multiselect("Select the preprocessing steps", options=steps)
        for step in selected_steps:
            self.preprocessor.add_step(PreProcessingStepsV2(step))

    def select_model_language(self):
        models = [e.value for e in SpacyModelNames]
        return st.sidebar.selectbox('Spacy model name', models)

    def preprocess_button(self, df, columns, model_name, include_token_count):
        self.preprocessor.set_model_name(model_name)
        if st.sidebar.button("Run Pre-processing"):
            with st.spinner('Pre-processing, this may take a while'):
                start_time = time.time()  # Start timer
                processed_df = self.preprocessor.preprocess_df(df, columns, include_token_count)
                end_time = time.time()  # End timer
                # Calculate elapsed time and display it
                elapsed_time = end_time - start_time

            st.success(f"Pre-processing took {elapsed_time:.2f} seconds.")
            return processed_df

        return df

    

    def get_csv_download_link(self, df: pd.DataFrame, filename: str = "processed_data.csv"):
        # Convert DataFrame to CSV
        csv = df.to_csv(index=False)
        # Convert CSV to bytes
        b64 = base64.b64encode(csv.encode()).decode()
        # Create and return download link
        return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download processed data as CSV</a>'

    def render(self):
        try:
            st.set_page_config(layout="wide")
        except:
            pass
        add_page_title()

        if self.homepage_processor.DATA not in st.session_state:
            st.sidebar.info('Configuration will be available here')
            st.error("It seems like no document is uploaded, you should probably visit the Homepage or upload a file bellow")
            uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
            data = self.homepage_processor.process_uploaded_file(uploaded_file)
            df = data
            st.button("Confirm")
            return
    
        df = st.session_state[self.homepage_processor.DATA]

        if df is not None and not df.empty:
            model_language = self.select_model_language()
            selected_columns = self.select_columns(df)
            self.select_preprocessing_steps()
            include_token_count = st.sidebar.checkbox('Include token count')
            processed_df = self.preprocess_button(df, selected_columns, model_language, include_token_count)
            st.dataframe(processed_df.head(), use_container_width=True)
            if processed_df is not None:
                download_link = self.get_csv_download_link(processed_df)
                st.sidebar.markdown(download_link, unsafe_allow_html=True)

if __name__ == "__main__":
    preprocessor = PreProcessingV2()
    homepage_processor = HomePageProcessor()
    ui = PreProcessingUI(preprocessor, homepage_processor)
    ui.render()
