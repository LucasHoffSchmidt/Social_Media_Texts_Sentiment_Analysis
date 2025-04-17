# Importing packages
import streamlit as st
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import BertTokenizer
import joblib

# Setting the page layout to wide to maximize size of wordcloud
st.set_page_config(layout="wide")

# App title
st.title("Social Media Texts Sentiment Analysis Interactive Dashboard")

# Loading and caching data
@st.cache_data
def load_data():
    return joblib.load("data/sentiment_text_data.pkl")

# Loading and caching resources
@st.cache_resource
def load_model():
    onnx_model_path = "model/tinybert_model_quantized.onnx"
    onnx_model = onnx.load(onnx_model_path)
    ort_session = ort.InferenceSession(onnx_model_path)
    tokenizer = BertTokenizer.from_pretrained("data/tinybert_tokenizer")
    
    return ort_session, tokenizer

df = load_data()
onnx_session, tokenizer = load_model()

# --- SECTION 1: Sentiment Filters ---
st.subheader("Select Sentiment Filters")

# Mapping labels to sentiments
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Creating radio sentiment buttons
selected_sentiment = st.radio("Choose sentiment:", ["Negative", "Neutral", "Positive"])

# Filtering dataframe
filtered_df = df[df["sentiment"] == selected_sentiment.lower()]

# --- SECTION 2: Wordcloud based on Filtering ---
st.subheader("Wordcloud for Selected Sentiments")

# Caching wordcloud to quickly update it on sentiment change
@st.cache_data
def generate_wordcloud(filtered_df):
    if not filtered_df.empty:
        text = " ".join(filtered_df["text"].astype(str).tolist())
        wordcloud = WordCloud(width=2000, height=1000, background_color="white").generate(text)
    
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        return fig
    else:
        return None

fig = generate_wordcloud(filtered_df)
if fig:
    st.pyplot(fig)
else:
    st.warning("No data available for this sentiment, please choose a different one")

# --- SECTION 3: Text Input Field for Custom Real-Time Sentiment Analysis ---
st.subheader("Get Sentiment for Custom Text")
custom_text = st.text_area("Enter your own text for sentiment analysis:", "")

# Creating functionality for button with the text "Analyse"
if st.button("Analyse"): # If the button is pressed
    if custom_text.strip(): # If the text is not just whitespace
        # Predicting sentiment with the ONNX model
        inputs = tokenizer(custom_text, return_tensors="pt", truncation=True, padding=True, max_length = 512)

        # Preparing inputs for ONXX model
        onnx_inputs = {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy()
        }

        # Running the ONNX model on the inputs and returning outputs
        onnx_outputs = onnx_session.run(None, onnx_inputs)
        logits = onnx_outputs[0]
        logits_tensor = torch.from_numpy(logits)

        # Getting probabilities for each sentiment class
        probs_tensor = F.softmax(logits_tensor, dim=1)
        probs = probs_tensor.numpy()

        # Getting the predicted label
        predicted_class = np.argmax(probs, axis=1).item()

        # Converting predicted label to sentiment class
        predicted_sentiment = label_map[predicted_class]

        # Getting the probability of the predicted sentiment class
        probability = probs[0][predicted_class]

        # Displaying prediction with color and probability
        if predicted_sentiment == "Positive":
            st.success(f"Prediction: {predicted_sentiment} ({probability:.2%} confidence)")
        elif predicted_sentiment == "Neutral":
            st.info(f"Prediction: {predicted_sentiment} ({probability:.2%} confidence)")
        else:
            st.error(f"Prediction: {predicted_sentiment} ({probability:.2%} confidence)")
    
        # Showing sentiment class probability as a bar
        st.progress(int(probability*100))
    else:
        st.warning("Please enter some text.")