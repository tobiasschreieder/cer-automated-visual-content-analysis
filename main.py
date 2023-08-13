from preprocessing.preprocessing import create_dataset, load_dataset, create_clarifai_dataset, load_clarifai_dataset
from evaluation.exploratory_data_analysis import run_exploratory_data_analysis
from evaluation.dataset_evaluation import run_dataset_evaluation, print_dataset_evaluation
# from computer_vision.computer_vision import run_clarifai
from classification.classification import Classifier
from sklearn.svm import SVC

"""
Main Method
"""

# Create dataset: topic_ids [51, 100] or set topic_ids = None for all possible topics;
# Specify size_dataset > 0 (images per topic)

# create_dataset(size_dataset=800)

# Load dataset as DataFrame for further use
# dataset = load_dataset()

# Exploratory Data Analysis
# run_exploratory_data_analysis()

# run_clarifai(image_ids=["I00000ed2c519e6b9710fd6ad", "I000a29953c54d0b4380724fa"])

#svm = SVC()
#clf = Classifier(svm, dataset, 51)
#clf.fit()
#clf.evaluate()

#run_dataset_evaluation('image_eval_t.txt', 'image_eval_p.txt')
print_dataset_evaluation()