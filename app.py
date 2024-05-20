import streamlit as st
import numpy as np
from transformers import AutoTokenizer, pipeline

import torch

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("ababio/Llama_3_8b_it_go_emotion")
    model = pipeline("text-generation", model="ababio/Llama_3_8b_it_go_emotion")
    return tokenizer, model


tokenizer, model = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

d = {
        0: 'admiration',
        1: 'amusement',
        2: 'anger',
        3: 'annoyance',
        4: 'approval',
        5: 'caring',
        6: 'confusion',
        7: 'curiosity',
        8: 'desire',
        9: 'disappointment',
        10: 'disapproval',
        11: 'disgust',
        12: 'embarrassment',
        13: 'excitement',
        14: 'fear',
        15: 'gratitude',
        16: 'grief',
        17: 'joy',
        18: 'love',
        19: 'nervousness',
        20: 'optimism',
        21: 'pride',
        22: 'realization',
        23: 'relief',
        24: 'remorse',
        25: 'sadness',
        26: 'surprise',
        27: 'neutral'
    }

if user_input and button :
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    st.write("Logits: ",output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    st.write("Prediction: ",d[y_pred[0]])