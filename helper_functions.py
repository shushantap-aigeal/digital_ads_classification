# All necessary imports
import pandas as pd
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
import pickle
import json
import os
import warnings

warnings.filterwarnings("ignore")
from io import StringIO
from nltk.corpus import stopwords


class Inferencing:

    """It accepts any type of input text or message preprocesses the text extracts features reduces its size and classifies the text"""

    def preprocessing(self, df):

        """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps.

        Input : Raw unpreprocessed text

        Output : Processed text removed html url stopwords blankspaces

        """

        import string, re

        # remove html if any present

        text = re.sub(r"http\S+", "", df)
        text = re.sub(r"Http\S+", "", text)

        text = nltk.word_tokenize(text)

        text = " ".join(text)

        # lowercasing of strings
        text = text.lower()

        # removing the articles in the text
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        text = re.sub(regex, " ", text)

        # removing the punctuation
        exclude = set(string.punctuation)
        text = "".join(ch for ch in text if ch not in exclude)

        # removing the white spaces
        text = " ".join(text.split())

        return text

    def feature_extract(self, cleaned_text, feature_extraction_model, svd_model):

        """Extract features from the data using TF-IDF vectorizer .

        Input : Cleaned Text for features extraction

        Output : Vectorized text with 5 dimensions

        """

        vectorized_output = feature_extraction_model.transform([cleaned_text])

        vectorized_output = vectorized_output.reshape(-1, 1)
        vectorized_output = svd_model.transform(vectorized_output.T)

        return vectorized_output

    def predict_text(self, cleaned_text, features, model):

        """Predicts model score for a single input data .

        Input : Cleaned Text & Features used for Prediction

        Output : Cleaned Text, Label_definition from 5 classes and score of the predicted output

        """

        output = dict()
        score = model.predict_proba(features)
        scores_list = score.tolist()
        categories = ["Banking", "Jobs-IT", "Rent-Apartment", "Retail", "Sell-House"]
        scores_table = dict(zip(categories, scores_list[0]))
        if (
            score[0][0] > score[0][1]
            and score[0][0] > score[0][2]
            and score[0][0] > score[0][3]
            and score[0][0] > score[0][4]
        ):
            score = round(score[0][0], 4)
        elif (
            score[0][1] > score[0][2]
            and score[0][1] > score[0][0]
            and score[0][1] > score[0][3]
            and score[0][1] > score[0][4]
        ):
            score = round(score[0][1], 4)
        elif (
            score[0][2] > score[0][1]
            and score[0][2] > score[0][0]
            and score[0][2] > score[0][3]
            and score[0][2] > score[0][4]
        ):
            score = round(score[0][2], 4)
        elif (
            score[0][3] > score[0][0]
            and score[0][3] > score[0][1]
            and score[0][3] > score[0][2]
            and score[0][3] > score[0][4]
        ):
            score = round(score[0][3], 4)
        else:
            score = round(score[0][4], 4)

        label = model.predict(features)
        if label[0] == 0:
            label_def = "Banking"
        elif label[0] == 1:
            label_def = "Jobs – IT"
        elif label[0] == 2:
            label_def = "Rent-Apartment"
        elif label[0] == 3:
            label_def = "Retail"
        else:
            label_def = "Sell-House"

        text = cleaned_text

        output = {
            "text": text,
            "label_definition": label_def,
            "score": score,
            "scores_table": scores_table,
        }
        return output
