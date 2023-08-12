from preprocessing.preprocessing import create_dataset, load_dataset, create_clarifai_dataset, load_clarifai_dataset
from evaluation.exploratory_data_analysis import run_exploratory_data_analysis
from computer_vision.computer_vision import run_clarifai


"""
Main Method
"""

# Create dataset: topic_ids [51, 100] or set topic_ids = None for all possible topics;
# Specify size_dataset > 0 (images per topic)
# create_dataset(topic_ids=[51, 52, 53, 54, 55], size_dataset=100)

# Load dataset as DataFrame for further use
# dataset = load_dataset()

# Exploratory Data Analysis
# run_exploratory_data_analysis()

# run_clarifai(image_ids=["I00000ed2c519e6b9710fd6ad", "I000a29953c54d0b4380724fa"])
