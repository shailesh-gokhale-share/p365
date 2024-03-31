# ~ ~ ~ ~ ~ ~ ~ ~ OM SHRI GANESHAAYA NAMAHA ~ ~ ~ ~ ~ ~ ~ ~ 

import streamlit as st
import joblib
import numpy as np
from scipy import sparse

sentiment_categories = {0: "V. Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4:"V. Positive"} 

# Load the trained model and preprocessing steps
model = joblib.load('lr_model.pkl')
# ordinal_encoder = joblib.load('customOrdinalEncoder.pkl')
tfidf_vectorizer = joblib.load('tfidf.pkl')

# App Title
st.title('Sentiment Analysis on Dell Latitude 7440 XCTO Reviews ')
st.markdown('Selected Model: Logistic Regression based on Vader Sentiment Compund Score')

# User input fields
rating = st.number_input('Enter Rating (1-5)', min_value=1, max_value=5)
title = st.text_input('Enter Title')
review = st.text_area('Enter Review')

# Preprocess user input. Apply the same preprocessing steps used during training
text_input = title + ' ' + review
text_tfidf = tfidf_vectorizer.transform([text_input])

rating_sparse = sparse.csr_matrix([[rating]])

# Create the input feature matrix
input_features = sparse.hstack([rating_sparse, text_tfidf])

# Make predictions
prediction = model.predict(input_features)
print(prediction)
predicted_sentiment = sentiment_categories.get(int(prediction[0]), "Unknown Category")

# #####################################################
# Display the predicted sentiment to the user
# #####################################################

# print(prediction[0])

# Define a function to generate color based on sentiment
def get_color(sentiment):
    # Define color gradient (red to green)
    colors = ['#FF0000', '#FF8000', '#FFBF00', '#80FF00', '#00FF00']
    # Define sentiment labels
    sentiments = ['V. Negative', 'Negative', 'Neutral', 'Positive', 'V. Positive']
    # Get index of sentiment label
    index = sentiments.index(sentiment)
    # Return corresponding color from gradient
    return colors[index]

# Get the color based on the value of the predicted_sentiment
color = get_color(predicted_sentiment)

# Display the predicted sentiment with dynamically assigned color
styled_text = '<p style="font-family: Arial; background-color: black; padding: 5px; color: {};">Predicted Sentiment: {}</p>'.format(color, predicted_sentiment)
st.markdown(styled_text, unsafe_allow_html=True)


# #################################################
# Test Cases
# #################################################

# Test case 1 (V. Positive)
# Rating: 5; Title: Great Laptop!; Review: Great laptop that I use to work from home. Screen size is perfect and love the flexibility to move around the house without dropping wifi. The only minor issue is for about a month, it would blue screen every so often (once every couple of weeks) but that seems to be resolved after applying updates. I get a lot of use out of this laptop and would recommend it.

# Test Case 2 (Negative)
# Rating: 1; Title: Bad Quality; Review: Bad Quality, overheating, and keyboard/trackpad issues.

# Test Case 3 (Negative)
# Rating: 2; Title: Many complications; Review: I received my new dell and it came with many complications and i still cant use it....software updates would not load up, programs refused to fully download on the computer that i needed for my job..and my job's IT dept has been working on it for two weeks now in order to make it a functional computer that i can rely on using. Hate to put a bad review but this is the truth

# Test Case 4 (Neutral)
# Rating: 2; Title: Dell Latitude 7440 XCTO; Review: Little flimsy and less ports. Screen is not very crisp. The touch screen was showing some issues right from the box

