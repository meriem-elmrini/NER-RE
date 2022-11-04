import streamlit as st


def define_sidebar(pages):
    with st.sidebar:
        page = st.sidebar.selectbox('Pages', pages)
        if page == pages[1]:
            model = st.radio('Choose your model',
                             ['syntax', 'tok2vec', 'transformer'])
            threshold = st.slider('Set your threshold',
                                  min_value=0.0,
                                  max_value=1.0,
                                  value=0.5,
                                  step=0.1)
        else:
            model = None
            threshold = None
    return page, model, threshold
