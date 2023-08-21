
import streamlit as st
from st_pages import Page, add_page_title, show_pages

class App:
    """
    Streamlit application for displaying pages and rendering content.

    Attributes:
        None
    """

    def __init__(self):
        """
        Initialize the App class and set up page configuration.

        Args:
            None

        Returns:
            None
        """
        show_pages(
            [
                Page("./homepage.py", "Homepage", "🏠"),
                Page("./preprocessingv2.py", "Pre-processingv2", "🧹"),
                Page("./topicmodelingtool.py", "Bertopic", "📚"),
                Page("./pygwalker_page.py", "PygwalkerPage", "📊")
            ]
        )

    def render(self) -> None:
        """
        Render the Streamlit application and display the initial page.

        Args:
            None

        Returns:
            None
        """
        try:
            st.set_page_config(layout="wide")
        except:
            pass
        add_page_title()
        st.info("You should start at the Homepage")

if __name__ == "__main__":
    App().render()