import streamlit as st
import numpy as np
import requests
from unsloth import FastLanguageModel
import torch
import nltk.data

max_seq_length = 2048
dtype = None
load_in_4bit = True

text_split_tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

def get_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="ababio/gemma-7b-it_go_emotion_NS1000",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map="cuda:0"  # Automatically map model parts to available devices
    )
    return tokenizer, model

tokenizer, model = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

d = {
    0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval', 5: 'caring',
    6: 'confusion', 7: 'curiosity', 8: 'desire', 9: 'disappointment', 10: 'disapproval', 11: 'disgust',
    12: 'embarrassment', 13: 'excitement', 14: 'fear', 15: 'gratitude', 16: 'grief', 17: 'joy',
    18: 'love', 19: 'nervousness', 20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief',
    24: 'remorse', 25: 'sadness', 26: 'surprise', 27: 'neutral'
}

alpaca_prompt = """Below is a conversation between a human and an AI agent. write a response based on the input.
### Instruction:
{}
### Input:
{}
### Response:
{}
"""

def predict_emotion(model, text):
    inputs = tokenizer(
        [alpaca_prompt.format(
            "Suggest a word or words that describes the emotion of the statement", 
            text, 
            ""
        )], return_tensors="pt", padding=True, truncation=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    input_length = len(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=input_length + 16,
            temperature=0.000001,
            num_beams=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
        predicted_ids = outputs[0, input_length:].cpu().numpy()
        
    predicted_emotion = tokenizer.decode(predicted_ids, skip_special_tokens=True)
    return predicted_emotion

if user_input and button:
    input_array = text_split_tokenizer.tokenize(user_input)

    predictions = []
    for sentence in input_array:
        emotion = predict_emotion(model, sentence)
        predictions.append(emotion)

    for i, sentence in enumerate(input_array):
        st.write(f"Sentence: {sentence}")
        st.write(f"Predicted Emotion: {predictions[i]}")