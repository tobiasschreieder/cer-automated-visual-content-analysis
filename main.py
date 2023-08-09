from preprocessing.preprocessing import create_dataset, load_dataset

"""
Main Method
"""

# Create dataset: topic_ids [51, 100] or set topic_ids = None for all possible topics;
# Specify size_dataset > 0 (images per topic)
create_dataset(topic_ids=[51, 52], size_dataset=10)

# Load dataset as DataFrame for further use
dataset = load_dataset()
