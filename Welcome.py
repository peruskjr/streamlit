import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to my Streamlit Dashboard! 👋")

    st.sidebar.success("Select an ml app above.")

    st.markdown(
        """
        If you've made it this far checking out my stuff, I greatly appreciate it!
        **👈 Select one of my ml apps from the sidebar** to see some examples
        of what I've been tinkering with using Streamlit!
    """
    )


if __name__ == "__main__":
    run()



