from dotenv import load_dotenv
import logging
import sys
import streamlit as st
from search_engine import get_query_engine
from html_template import css


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    
def main():
    load_dotenv()
    query_engine = get_query_engine()
    st.set_page_config(page_title="Ask doubts about Operating Systems:", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header('Ask doubts about Operating Systems :books:')
    user_question = st.text_input("Ask a question about operating systems: ")
    if user_question or st.button("Submit"):
         with st.spinner("Processing..."):
            st.write(query_engine.query(user_question).response)
            st.write(query_engine.query(user_question).source_nodes[0].metadata['file_name'])

if __name__ == "__main__":
    main()