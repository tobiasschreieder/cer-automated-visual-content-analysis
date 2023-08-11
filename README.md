# CER - Automated Visual Content Analysis

## Dataset

Please download the touche23-image-search-main.zip, topics.xml from https://zenodo.org/record/7628213

Save unzipped dataset in data/touche23-image-search-main and topics under data/topics.xml


## Annotation
Start annotation server with "python start_annotation.py -web". Please append "/evaluation" to your local IP address.
The annotation results will be saved at "working/image_eval.csv".


## Clarifai Computer Vision
For Clarifai API usage please save Personal Access Token (PAT) at working/pat.txt. The Clarifai model can be specified in computer_vision.computer_vision.py (Default: MODEL_ID = 'general-image-recognition').