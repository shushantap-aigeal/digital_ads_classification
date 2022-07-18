import streamlit as st
from helper_functions import Inferencing
from config import model, feature_extraction_model, svd_model, data_file_path
import pandas as pd
import random

data = pd.read_csv(data_file_path)
data.dropna(inplace=True)
data.reset_index(drop=True)

st.header("Advertisment Classifier")

text = st.text_area(" ", height=200, placeholder=None)
inference_obj = Inferencing()

if st.button("PREDICT"):

    cleaned_text = inference_obj.preprocessing(text)
    features = inference_obj.feature_extract(
        cleaned_text, feature_extraction_model, svd_model
    )
    predicted_output = inference_obj.predict_text(cleaned_text, features, model)
    predicted_score_categories = predicted_output["scores_table"]

    # st.write("**INPUT TEXT** :", predicted_output["text"])
    score = round(predicted_output["score"]*100)
    st.success(f'{predicted_output["label_definition"]} - {score}')
    st.progress(score)
    st.write("**SCORES FOR EACH CATEGORY** :")

    st.json(predicted_score_categories)
    predicted_label = predicted_output["label_definition"]
    
    if predicted_label:
        select_data = data[data.JobType == predicted_label]
        st.subheader("Similar Digital Advertisments")
        select_data = select_data[["title"]][:5]
        select_data = select_data.reset_index(drop=True)
        st.dataframe(select_data)
    else:
        st.write("No Similar Advertisments to Recommend")

else:
    st.write("Press predict button to predict")
