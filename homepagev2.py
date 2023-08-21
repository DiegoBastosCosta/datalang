import pandas as pd
import streamlit as st
from st_pages import add_page_title

class HomePageProcessor:
    DATA = 'data'

    def process_uploaded_file(self, uploaded_file):
        if uploaded_file is not None:
            if uploaded_file.type == "text/csv":
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file, sheet_name='Relatorio')
            st.session_state[HomePageProcessor.DATA] = data
            return data
        return None

class HomePageUI:
    def __init__(self, processor: HomePageProcessor):
        self.processor = processor

    def render(self):
        try:
            st.set_page_config(layout="wide")
        except:
            pass
        add_page_title()

        st.header("Welcome to Datalang")
        st.caption("This tool is part of a data analysis pipeline that aims to democratize artificial intelligence")
        st.sidebar.write("Datalang is an open source tool made by Bruno Brandão Borges as a learning and sharing experiment on artificial intelligence")
        st.write('---')

        if HomePageProcessor.DATA not in st.session_state:
            st.info("You should start by uploading the base document you want to work with")
            st.caption("This document will be used and transformed across the application's pipeline to provide new representations and insight")
            uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
            data = self.processor.process_uploaded_file(uploaded_file)
        else:
            st.warning("You already uploaded your document, if you wish to start a new work, save the current state and reload the page")
            data = st.session_state[HomePageProcessor.DATA]
            min_value = min(1, len(data))
            max_value = max(50, len(data))
            sample_size = st.slider('Sample size', min_value=min_value, max_value=max_value, value=int(max_value/2))
            st.dataframe(data.sample(n=sample_size), use_container_width=True)
            st.info("It's recommended that you continue to the pre-processing page")
        st.button("Continue")

if __name__ == "__main__":
    processor = HomePageProcessor()
    ui = HomePageUI(processor)
    ui.render()
