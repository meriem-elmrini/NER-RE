import streamlit as st
import numpy as np

from settings import PAGES
from navigation_bar import define_sidebar
from dashboard.home import write_home
from dashboard.try_model import define_inputs, get_prediction, get_spans, display_spans


def main():
    st.title('Patent - Relation extraction')
    page, model, threshold = define_sidebar(PAGES)
    if page == PAGES[0]:
        write_home()
    if page == PAGES[1]:
        sentence, type_spans, entered_spans = define_inputs()
        if st.button('Get predictions'):
            spans, predicted_relations = get_prediction(model,
                                                        sentence,
                                                        type_spans,
                                                        entered_spans,
                                                        threshold=threshold)
            st.markdown('**Predictions :**')
            if spans and not type_spans:
                display_spans(sentence, get_spans(sentence, [span[1] for span in spans]))
            if predicted_relations:
                for rel in predicted_relations:
                    if len(rel) != 0:
                        if len(rel[0]) > 3:
                            st.write(rel[0][0], '→', rel[0][2], '→', rel[0][1], np.round(rel[0][3], 2))
                        else:
                            st.write(rel[0][0], '→', rel[0][2], '→', rel[0][1])
                    else:
                        st.write('**No relations detected.**')
            else:
                st.write('**No relations detected.**')


if __name__ == '__main__':
    main()
