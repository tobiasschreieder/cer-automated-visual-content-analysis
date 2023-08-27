from preprocessing.preprocessing import load_dataset
from evaluation.exploratory_data_analysis import run_exploratory_data_analysis
from evaluation.dataset_evaluation import run_dataset_evaluation, print_dataset_evaluation
from classification.classification import Classifier

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier

import json

"""
Main Method
"""

#########################################

# This file is ready to run
# Note that the grid searching process can take a longer time.
# (11 hours on the system we used)

#########################################

# Exploratory Data Analysis
run_exploratory_data_analysis()

# evaluation of data set coding
run_dataset_evaluation('image_eval_a.txt', 'image_eval_b.txt')
print_dataset_evaluation()


# load models for grid search
with open('classification/models.json', 'r') as f:
    models = json.load(f)

# load or create datasets
data = {
    # clariafai
    'dataset_clarifai_topics=55+76+81_size=900.pkl': load_dataset(size_dataset=900, topic_ids=[55,76,81], use_clarifai_data=True),

    # combined
    'dataset_combined_topics=55+76+81_size=900.pkl': load_dataset(size_dataset=900, topic_ids=[55,76,81], use_combined_dataset=True),

    # google
    'dataset_touche_topics=55+76+81_size=900.pkl': load_dataset(size_dataset=900, topic_ids=[55,76,81]),
    # five-topic-case
    'dataset_touche_topics=51+55+76+81+100_size=900.pkl': load_dataset(size_dataset=900,  topic_ids=[51,55,76,81,100])
}

# for all defined models...
for key in models.keys():
    # ... and all defined data sets
    for file in data.keys():

        # get all topic_ids for processing
        topics = data[file]['topic_id'].unique()

        # init classifiers
        if key == 'svm':
            model = SVC()
        elif key == 'gbc':
            model = GradientBoostingClassifier()
        elif key == "sgd":
            model = SGDClassifier()
        elif key == 'pac':
            model = PassiveAggressiveClassifier()

        # parameters to be used in the grid search
        grid_params = models[key]

        # init and run the classifier/grid search for every topic
        for topic in topics:

            print(f'Start performing BINARY grid search for {file}\nTopic: {topic} out of {list(topics)}\nClassifier: {str(model)}')
            # don't use likelihood
            clf = Classifier(model, data[file], topic, grid_params=grid_params, dataset_name=file, use_likelihood=False)
            clf.evaluate_grid_search()

            print(f'Start performing LIKELIHOOD grid search for {file}\nTopic: {topic} out of {list(topics)}\nClassifier: {str(model)}')
            # use likelihood
            clf = Classifier(model, data[file], topic, grid_params=grid_params, dataset_name=file, use_likelihood=True)
            clf.evaluate_grid_search()

# The results can be evaluated with evaluation_of_result.ipynb in /evaluation
