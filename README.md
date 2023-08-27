# CER - Automated Visual Content Analysis

## Dataset

Please download the touche23-image-search-main.zip, topics.xml from https://zenodo.org/record/7628213

Save unzipped dataset in data/touche23-image-search-main and topics under data/topics.xml


## Annotation
Start annotation server with "python start_annotation.py -web". Please append "/evaluation" to your local IP address.
The annotation results will be saved at "working/image_eval.csv".


## Clarifai Computer Vision
For Clarifai API usage please save Personal Access Token (PAT) at working/pat.txt. The Clarifai model can be specified in computer_vision.computer_vision.py (Default: MODEL_ID = 'general-image-recognition').

## Classification
The classifier class can be used to carry out the training and evaluation of classifiers, as well as performing and evaluating a grid search on a specified calssifier. The class was specially designed to handle data sets of computer vision generated image labels, like they can be generated with the preprocessing module of this project.
The evaluation has an option to save results to a JSON-file. Generated JSON documents contain all relevant information about the classifier, like name, parameters, etc. as well as relevant evaluation metrics.

An example for the use of the class can be seen below:

```
data = load_dataset(size_dataset=900, topic_ids=[55,76,81])
clf = Classifier(SVC(), data, topic_id=55, grid_params=grid_params)
clf.evaluate_grid_search()
```

Where `SVC()` is the passed classifier, `load_dataset` is from the preprocessing module and creates the dataset, `topic_id` is one of the topic ids present in `data` and `grid_params` is a valid parameter configuration of the specified classifier (which in this case is the default Support Vector Classifier from Scikit-Learn). For further use, refer to the class documentation.

In this project, the used configurations for the grid search are defined in the`models.json` file.


## Evaluation
Different evaluation tasks can be performed within this project.
`dataset_evaluation.py` evaluates the annotations of the Touché23 data set by the two authors and saves them as CSV-files. The same evaluations are additionally available in a notebook format. `exploratory_data_analysis.py` creates a markdown file with an extensive exploratory analysis of the outputs of the used computer vision APIs. `evaluation_of_results.ipynb` contains the evaluations of the tested model outputs in a notebook format.