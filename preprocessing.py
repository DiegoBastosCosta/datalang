import spacy
import pandas as pd
from enum import Enum
import streamlit as st
from spacy.cli import download
from st_pages import add_page_title

class PreProcessingSteps(Enum):
    STOP_WORDS_REMOVAL = 'stop words removal'
    LEMMATIZATION = 'lemmatization'

class PreProcessingPage:
    """
    The pre-processing page of the Datalang data analysis pipeline.

    This class represents the pre-processing page of the Streamlit application. It provides various text pre-processing
    functionalities such as stop words removal and lemmatization. Users can select the processing steps, configure the
    pipeline, and apply the pre-processing to the chosen text columns in the uploaded data.

    Attributes:
        DATA (str): A constant representing the key used to store the uploaded data in the Streamlit session state.
        COLUMN_LEMMA (str): A constant suffix added to the column name for the lemmatized version of the text.
        COLUMN_NO_STOP (str): A constant suffix added to the column name for the text with stop words removed.
        COLUMN_PROCESSED (str): A constant suffix added to the column name for the pre-processed version of the text.
        COLUMN_TOKEN_COUNT (str): A constant suffix added to the column name for the token count of the text.
        SPACY_MODEL_NAME (str): A constant representing the key used to store the selected Spacy language model name.
        CHOSEN_PROCESSING_STEPS (str): A constant representing the key used to store the selected processing steps.
        SPACY_LANGUAGE_OPTIONS (List[str]): A list of available Spacy language model names.

    Methods:
        __init__(): Initializes the PreProcessingPage object and loads the Spacy language model.
        _load_spacy_model(): Loads the selected Spacy language model.
        count_tokens(text: str) -> int: Counts the number of tokens in the given text.
        remove_stop_words(text: str) -> str: Removes stop words from the given text.
        lemmatization(text: str) -> str: Performs lemmatization on the given text.
        preprocess_text(text: str, step: PreProcessingSteps) -> str: Applies the specified pre-processing step to the given text.
        generate_preprocess_step(step: PreProcessingSteps, df: pd.DataFrame, columns: list) -> pd.DataFrame:
            Generates a new DataFrame with columns containing the pre-processed text based on the specified step.
        preprocessing_controller(text: str) -> str: Applies all the chosen pre-processing steps to the given text.
        render_preprocessing_steps(data: pd.DataFrame, columns_to_process: list) -> None:
            Renders the UI elements for configuring and applying the pre-processing steps.
        preprocess_chosen_steps(df: pd.DataFrame, columns: list, include_token_count: bool) -> pd.DataFrame:
            Pre-processes the selected columns in the DataFrame based on the chosen processing steps.
        get_chosen_processing_steps() -> List[PreProcessingSteps]: Returns the selected pre-processing steps.
        render() -> None: Renders the pre-processing page UI.

    """

    DATA = 'data'
    COLUMN_LEMMA = '_lemma'
    COLUMN_NO_STOP = '_no_stop'
    COLUMN_PROCESSED = '_processed'
    COLUMN_TOKEN_COUNT = '_token_count'
    SPACY_MODEL_NAME = 'spacy_model_name'
    CHOSEN_PROCESSING_STEPS = 'chosen_preprocessing_steps'
    SPACY_LANGUAGE_OPTIONS = ['en_core_web_sm', 'xx_ent_wiki_sm']

    def __init__(self):
        """
        Initializes the PreProcessingPage object and loads the Spacy language model.
        """
        st.session_state[self.CHOSEN_PROCESSING_STEPS] = st.session_state.get(self.CHOSEN_PROCESSING_STEPS, [])
        self.spacy_model = self._load_spacy_model()

    def _load_spacy_model(self) -> spacy.Language:
        """
        Loads the selected Spacy language model.

        Returns:
            spacy.Language: The loaded Spacy language model.
        """
        model_name = st.session_state.get(self.SPACY_MODEL_NAME, self.SPACY_LANGUAGE_OPTIONS[0])
        try:
            return spacy.load(model_name)
        except OSError:
            st.warning(f"The model {model_name} is not installed. Attempting to download...")
            download(model_name)
            st.success(f"Model {model_name} downloaded successfully.")
            return spacy.load(model_name)
    
    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in the given text.

        Args:
            text (str): The input text.

        Returns:
            int: The number of tokens in the text.
        """
        if not isinstance(text, str):
            return None
        doc = self.spacy_model(text)
        return len(doc)
    
    def remove_stop_words(self, text: str) -> str:
        """
        Removes stop words from the given text.

        Args:
            text (str): The input text.

        Returns:
            str: The text with stop words removed.
        """
        if not isinstance(text, str):
            return text
        doc = self.spacy_model(text)
        return ' '.join([token.text for token in doc if not token.is_stop])

    def lemmatization(self, text: str) -> str:
        """
        Performs lemmatization on the given text.

        Args:
            text (str): The input text.

        Returns:
            str: The lemmatized text.
        """
        if not isinstance(text, str):
            return text
        doc = self.spacy_model(text)
        return ' '.join([token.lemma_ for token in doc])
        
    def preprocess_text(self, text: str, step: PreProcessingSteps) -> str:
        """
        Applies the specified pre-processing step to the given text.

        Args:
            text (str): The input text.
            step (PreProcessingSteps): The pre-processing step to apply.

        Returns:
            str: The pre-processed text.
        """
        if step == PreProcessingSteps.STOP_WORDS_REMOVAL:
            return self.remove_stop_words(text)
        elif step == PreProcessingSteps.LEMMATIZATION:
            return self.lemmatization(text)
        else:
            return text

    def generate_preprocess_step(self, step: PreProcessingSteps, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Generates a new DataFrame with columns containing the pre-processed text based on the specified step.

        Args:
            step (PreProcessingSteps): The pre-processing step to apply.
            df (pd.DataFrame): The input DataFrame.
            columns (list): The columns to process.

        Returns:
            pd.DataFrame: The resulting DataFrame with pre-processed columns.
        """
        result = pd.DataFrame()
        for col in columns:
            result[col] = df[col]
            result[col + self.COLUMN_PROCESSED] = df[col].apply(lambda text: self.preprocess_text(text, step))
        return result
    
    def preprocessing_controller(self, text: str) -> str:
        """
        Applies all the chosen pre-processing steps to the given text.

        Args:
            text (str): The input text.

        Returns:
            str: The pre-processed text.
        """
        for step in self.get_chosen_processing_steps():
            text = self.preprocess_text(text, step)
        return text

    def render_preprocessing_steps(self, data: pd.DataFrame, columns_to_process: list) -> None:
        """
        Renders the UI elements for configuring and applying the pre-processing steps.

        Args:
            data (pd.DataFrame): The input DataFrame.
            columns_to_process (list): The selected columns to process.

        Returns:
            None
        """
        st.subheader("Pipeline configuration")
        for step in PreProcessingSteps:
            with st.expander(step.value):
                if st.checkbox("View and apply", value=step in st.session_state[self.CHOSEN_PROCESSING_STEPS], key=f'apply_{step}'):
                    if step not in st.session_state[self.CHOSEN_PROCESSING_STEPS]:
                        st.session_state[self.CHOSEN_PROCESSING_STEPS].append(step)
                    preprocessed_data = self.generate_preprocess_step(step, data, columns_to_process)
                    st.dataframe(preprocessed_data, use_container_width=True)
                else:
                    if step in st.session_state[self.CHOSEN_PROCESSING_STEPS]:
                        st.session_state[self.CHOSEN_PROCESSING_STEPS].remove(step)
 
    def preprocess_chosen_steps(self, df: pd.DataFrame, columns: list, include_token_count: bool) -> pd.DataFrame:
        """
        Pre-processes the selected columns in the DataFrame based on the chosen processing steps.

        Args:
            df (pd.DataFrame): The input DataFrame.
            columns (list): The columns to process.
            include_token_count (bool): Whether to include token count columns.

        Returns:
            pd.DataFrame: The pre-processed DataFrame.
        """
        for col in columns:
            df[col + '_processed'] = df[col].apply(self.preprocessing_controller)

        if include_token_count:
            string_columns = [col for col in df.columns if df[col].dtype == 'object']
            for col in string_columns:
                df[col + '_token_count'] = df[col].apply(self.count_tokens)

        return df
    
    def get_chosen_processing_steps(self) -> list:
        """
        Returns the selected pre-processing steps.

        Returns:
            list: The selected pre-processing steps.
        """
        return st.session_state.get(self.CHOSEN_PROCESSING_STEPS, [])

    def render(self) -> None:
        """
        Renders the pre-processing page UI.

        Returns:
            None
        """
        try:
            st.set_page_config(layout="wide")
        except:
            pass
        add_page_title()

        if self.DATA not in st.session_state:
            st.error("It seems like no document is uploaded, you should probably visit the Homepage")
            return
        
        data = st.session_state[self.DATA]
        sample_size = st.sidebar.slider("Sample Size", min_value=1, max_value=int(0.1*len(data)), value=int(0.01*len(data)), step=1)
        sample_data = data.sample(n=sample_size)

        st.caption("Over here you can pre-process your data, which is highly recommended before continuing")
        st.write("Sample data")
        st.dataframe(sample_data, use_container_width=True)

        string_columns = [column for column in data.columns if data[column].dtype == 'object']
        selected_columns = st.sidebar.multiselect(
            "Select the text columns to process",
            options=string_columns
        )

        st.session_state[self.SPACY_MODEL_NAME] = st.sidebar.selectbox(
            'Select a language model',
            options=self.SPACY_LANGUAGE_OPTIONS
        )

        st.write('---')
        self.render_preprocessing_steps(sample_data, selected_columns)

        include_token_count = st.sidebar.checkbox('Include Token Count', value=False)
        if st.sidebar.button("Run Pre-processing"):
            with st.spinner("Pre processing... This could take a while"):
                st.session_state[self.DATA] = self.preprocess_chosen_steps(data, selected_columns, include_token_count)
            st.write('---')
            st.success("Columns pre-processed successfully")
            st.write('resulting sample')
            st.dataframe(st.session_state[self.DATA].sample(n=sample_size), use_container_width=True)

if __name__ == "__main__":
    PreProcessingPage().render()
