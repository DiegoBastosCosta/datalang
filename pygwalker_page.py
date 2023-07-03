import streamlit as st
import pandas as pd
import pygwalker as pyg
import streamlit.components.v1 as components
from st_pages import add_page_title

class PygwalkerPage:
    DATA = 'data'
    TOPIC_MODELING_RESULT = 'topic_modeling_result'

    def render(self):
        try:
            st.set_page_config(
                page_title="Use Pygwalker In Streamlit",
                layout="wide"
            )
        except:
            pass

        add_page_title()

        available_dataframes = []
        if self.DATA in st.session_state:
            available_dataframes.append(self.DATA)
        if self.TOPIC_MODELING_RESULT in st.session_state:
            available_dataframes.append(self.TOPIC_MODELING_RESULT)

        if not available_dataframes:
            st.error("It seems like no document is uploaded, you should probably visit the Homepage")
            return

        selected_dataframes = st.multiselect('Select the dataframes you want to use on this pipeline', available_dataframes)

        for df_name in selected_dataframes:
            data = st.session_state[df_name]

            with st.expander(f"Loaded data - {df_name} - 100 first out of {len(data)}"):
                st.dataframe(data[0:200], use_container_width=True)

            pyg_html = pyg.walk(data, return_html=True)

            with st.expander(f"Pygwalker visualization - {df_name}"):
                components.html(pyg_html, height=1000, scrolling=True)

if __name__ == "__main__":
    PygwalkerPage().render()
