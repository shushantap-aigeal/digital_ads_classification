import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")

# get current working directory
cwd = os.getcwd()
path = os.path.join(cwd, "models/")
data_path = os.path.join(cwd, "data/")


# uncomment this to use the version one model while comment the version 2 model
# name_of_main_model = "finalized_model_XGB.sav"
# name_of_feature_model = "vectorizer.sav"
# name_of_feature_reducer_model = "svd.sav"


# uncomment this to use the version two model
name_of_main_model = "finalized_model_XGBV2.sav"
name_of_feature_model = "vectorizerV2.sav"
name_of_feature_reducer_model = "svdV2.sav"


# provide the paths
main_model_path = path + name_of_main_model
feature_model_path = path + name_of_feature_model
dimensionality_reducer = path + name_of_feature_reducer_model


# load the models
model = pickle.load(open(main_model_path, "rb"))
feature_extraction_model = pickle.load(open(feature_model_path, "rb"))
svd_model = pickle.load(open(dimensionality_reducer, "rb"))


# get source data for display
name_of_the_file = "concatenated_data.csv"
data_file_path = data_path + name_of_the_file
