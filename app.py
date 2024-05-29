import streamlit as st
import numpy as np
import nltk
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import os
import re
import tempfile

# Function to get credentials from environment variable and create a temporary file
def get_credentials():
    creds_json_str = os.getenv("JSONSTR")  # Get JSON credentials stored as a string
    if creds_json_str is None:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_JSON not found in environment")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as temp:
        temp.write(creds_json_str)  # Write in JSON format
        temp_filename = temp.name 

    return temp_filename

# Set environment variable for Google application credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = get_credentials()

max_seq_length = 2048
dtype = None
load_in_4bit = True

# Check if 'punkt' is already downloaded, otherwise download it
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

text_split_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Function to predict emotions using the custom trained model
def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-east4",
    api_endpoint: str = "us-east4-aiplatform.googleapis.com",
) -> List[str]:
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    predictions_list = []
    predictions = response.predictions
    for prediction in predictions:
        if isinstance(prediction, str):
            clean_prediction = re.sub(r'(\n|Origin|###|Optimization|Response:)', '', prediction)
            split_predictions = clean_prediction.split()
            predictions_list.extend(split_predictions)
        else:
            print(" prediction (unknown type, skipping):", prediction)
    return [emotion for emotion in predictions_list if emotion in d_emotion.values()]

d_emotion = {0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval', 5: 'caring', 6: 'confusion',
             7: 'curiosity', 8: 'desire', 9: 'disappointment', 10: 'disapproval', 11: 'disgust', 12: 'embarrassment',
             13: 'excitement', 14: 'fear', 15: 'gratitude', 16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness',
             20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief', 24: 'remorse', 25: 'sadness', 26: 'surprise',
             27: 'neutral'}

st.write(" ")
st.write(" ")
st.header('Sentiment: Emotion Analyses', divider='rainbow')
st.write('Provide any number of document texts to analyse the emotion percentages')


# # Define the sample text
# sample_text = ("Once, in a small village nestled in the rolling hills of Tuscany, lived an elderly woman named Isabella. "
#                "She had spent her entire life in this village, raising her children and caring for her garden, which was the most "
#                "beautiful in the region. Her husband, Marco, had passed away many years ago, leaving her with a heart full of memories "
#                "and a small, quaint house that overlooked the lush vineyards.")

# # Add button to fill in sample text
# if st.button("Use Sample Text"):
#     user_input = st.text_input(label="sample", value=sample_text, label_visibility="hidden")
# else:

# option = st.selectbox(
#     "How would you like to be contacted?",
#     ("i hate that food", "happy day", "Mobile phone"))

# st.write("You selected:", option)


optionValue = ""
user_input = ""

# st.write("Select sample texts analyse emotion 👉")
# village = st.checkbox("Sample 1")
# life = st.checkbox("Sample 2")

option = st.selectbox(
   "Select sample texts to analyse emotion 👉 :sunglasses: :smile: :angry: :disappointed: :fearful: :rage:  :weary:	:cry: :sweat_smile: :neutral_face: :blush: :heart_eyes: :innocent: :satisfied: :joy:",
   (
       "Sample 1: Getting late for a meeting", 
       "Sample 2: I consider myself an observant"
   ),
   index=None,
   placeholder="Select...",
)


if option ==  "Sample 1: Getting late for a meeting":
    optionValue = """Getting late for a meeting, need to run’, he said, as he slung his coat over the shoulder, and bounded out of the house. As he drove away, she came running down the stairs two at a time. ‘Wait, wait’, she said, but he had already left.
Her mouth crumpled like used wrapping paper. ‘He forgot to give me a goodbye kiss’, she whispered in a voice that trembled under the weight of her hurt. She called him, ‘you left without giving me a kiss’, she said accusingly. ‘I am sorry sweetheart’, he said, his voice contrite. ‘It is okay’, she said, trying to be all grown up as she cut the call.
She gulped down her breakfast morosely, wore her shoes, picked up her school bag and started to walk out of the door, her shoulders slumped. As she climbed down the steps, the car glided to a stop outside the house. He got out of the car. She ran to him, her whole face lit up like a Christmas tree.
‘I am sorry I forgot’, he said, as he picked her up and hugged her. She said nothing. Her jaw ached from smiling.
Fifteen years later, no one would remember he was late for a meeting, but a little girl would never ever forget that her father drove all the way back home just to kiss her goodbye!
III. Lesson For Everyone
A son took his old father to a restaurant for an evening dinner. Father being very old and weak, while eating, dropped food on his shirt and trousers. Others diners watched him in disgust while his son was calm.
After he finished eating, his son who was not at all embarrassed, quietly took him to the wash room, wiped the food particles, removed the stains, combed his hair and fitted his spectacles firmly. When they came out, the entire restaurant was watching them in dead silence, not able to grasp how someone could embarrass themselves publicly like that.
The son settled the bill and started walking out with his father.
At that time, an old man amongst the diners called out to the son and asked him, “Don’t you think you have left something behind?”.
The son replied, “No sir, I haven’t”.
The old man retorted, “Yes, you have! You left a lesson for every son and hope for every father“.
IV. The Sweeter Apple
A lovely little girl was holding two apples in her hands.
Her mom came in and softly asked her little daughter with a smile, “My sweetie, could you give your mom one of your two apples?”
The girl looked up at her mom for some seconds, then she suddenly took a quick bite on one apple, and then quickly on the other.
The mom felt the smile on her face freeze. She tried hard not to reveal her disappointment.
Then the little girl handed one of her bitten apples to her mom, and said, “Mommy, here you are. This is the sweeter one.”
V. Without Your Love, I Would Die
5 Emotional Short Stories That Will Make You Cry Insantly (1).png
One day a boy and a girl were driving home after watching a movie. The boy sensed there was something wrong because of the painful silence they shared between them that night.
The girl then asked the boy to pull over because she wanted to talk. She told him that her feelings had changed and that it was time to move on.
A silent tear slid down his cheek as he slowly reached out to his pocket and passed her a folded note.
At that moment, a drunk driver was speeding down that very same street. He severed right into the car killing the boy.
Miraculously, the girl survived. Remembering the note later, she unfolded it and read,
‘Without your love, I would die.’"""
elif option == "Sample 2: I consider myself an observant": 
    optionValue = """I consider myself an observant. I often spend time being quiet, just staring and watching my surroundings, from the main situation to things that most people don’t notice, like that single fly who broke its wings and is currently struggling to turn its body over; like the dust piling in the corner of my bus’ automatic door.
During my bus ride home, I usually listen to some music from my Spotify (give it a follow if you want to) and just stare out the window, watching the city at night, admiring all its mesmerizing skyscrapers and the dim street lights.
It’s beautiful.
It made me realize that I have so many little things to be grateful for.
When I graduated, I had a dream of working in the middle of this busy city because I want to be able to experience Jakarta at night. Here I am.
I dreamed of working in a tall building so I can watch the streets from above. I didn’t get the experience at my previous office, but here I am now on the 18th floor (not that high, but still, I’m blessed.)
I dreamed of working in a well-known company. It didn’t come true during my first year of work, but here I am now.
I dreamed of going on a leisure trip with my friends, now I have gone to 2 different cities with them.
There are still so many other dreams that came true. More and more are coming true as time goes by. I cannot be more thankful.
Apparently life is not as boring if we appreciate the smallest things."""
else:
     optionValue =""
    
st.write(' ')
st.write(' ')

user_input = st.text_area('Enter or Paste Text to Analyze', value=optionValue)


button = st.button("Analyze")


if button and user_input:
    alpaca_prompt = """Below is a conversation between a human and an AI agent. write a response based on the input.
        ### Instruction:
        predict the emotion word or words
        ### Input:
        {}
        ### Response:
        """
    instances = []
    input_array = text_split_tokenizer.tokenize(user_input)
    for sentence in input_array:
        formatted_input = alpaca_prompt.format(sentence.strip())
        instance = {
            "inputs": formatted_input,
            "parameters": {
                "max_new_tokens": 4,
                "temperature": 0.00001,
                "top_p": 0.9,
                "top_k": 10
            }
        }
        instances.append(instance)

    predictions = predict_custom_trained_model_sample(
        project=os.environ["project"],
        endpoint_id=os.environ["endpoint_id"],
        location=os.environ["location"],
        instances=instances
    )

    emotion_counts = pd.Series(predictions).value_counts(normalize=True).reset_index()
    emotion_counts.columns = ['Emotion', 'Percentage']
    emotion_counts['Percentage'] *= 100  # Convert to percentage
    fig_pie = px.pie(emotion_counts, values='Percentage', names='Emotion', title='Percentage of Emotions in Given Text')
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')

    @st.cache_data
    def get_emotion_chart(predictions):
        emotion_counts = pd.Series(predictions).value_counts().reset_index()
        emotion_counts.columns = ['Emotion', 'Count']
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=emotion_counts['Emotion'],
            y=emotion_counts['Count'],
            marker_color='indianred'
        ))
        fig_bar.update_layout(title='Count of Each Emotion in Given Text', xaxis_title='Emotion', yaxis_title='Count')
        return fig_bar

    fig_bar = get_emotion_chart(predictions)

    @st.cache_data
    def get_emotion_heatmap(predictions):
        emotion_counts = pd.Series(predictions).value_counts().reset_index()
        emotion_counts.columns = ['Emotion', 'Count']
        
        heatmap_matrix = pd.DataFrame(0, index=d_emotion.values(), columns=d_emotion.values())
        for index, row in emotion_counts.iterrows():
            heatmap_matrix.at[row['Emotion'], row['Emotion']] = row['Count']
    
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_matrix.values,
            x=heatmap_matrix.columns.tolist(),
            y=heatmap_matrix.index.tolist(),
            text=heatmap_matrix.values,
            hovertemplate="Count: %{text}",
            colorscale='Viridis'
        ))
        fig.update_layout(title='Emotion Heatmap', xaxis_title='Predicted Emotion', yaxis_title='Predicted Emotion')
        return fig
        
    fig_heatmap = get_emotion_heatmap(predictions)
    
    # tab1, tab2, tab3 = st.tabs(["Emotion Analysis", "Emotion Counts Distribution", "Heatmap"])
    tab1, tab2 = st.tabs(["Emotion Analysis", "Emotion Counts Distribution"])
    with tab1:
        st.plotly_chart(fig_pie)
    with tab2:
        st.plotly_chart(fig_bar)
    # with tab3:
    #     st.plotly_chart(fig_heatmap)