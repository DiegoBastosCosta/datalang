import pandas as pd
import streamlit as st
from st_pages import add_page_title

class HomePage:
    """The homepage of the Datalang data analysis pipeline.

    This class represents the initial page of the Streamlit application. It provides a user interface for uploading a CSV
    file, which serves as the base document for further analysis and transformations in the Datalang pipeline.

    Attributes:
        DATA (str): A constant representing the key used to store the uploaded data in the Streamlit session state.

    Methods:
        render(): Renders the homepage UI, allowing users to upload a CSV file and proceed to the next step in the pipeline.
    """
    DATA = 'data'

    def render(self) -> None:
        try:
            st.set_page_config(layout="wide")
        except:
            pass
        add_page_title()

        st.header("Welcome to Datalang")
        st.caption("This tool is part of a data analysis pipeline that aims to democratize artificial intelligence")
        st.sidebar.write("Datalang is an open source tool made by Bruno Brandão Borges as a learning and sharing experiment on artificial intelligence")
        st.write('---')

        if HomePage.DATA not in st.session_state:
            st.info("You should start by uploading the base document you want to work with")
            st.caption("This document will be used and transformed across the application's pipeline to provide new representations and insight")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                st.session_state[HomePage.DATA] = data
        else:
            st.warning("You already uploaded your document, if you wish to start a new work, save the current state and reload the page")
            st.info("It's recommended that you continue to the pre-processing page")
        st.button("Continue")


if __name__ == "__main__":
    HomePage().render()
