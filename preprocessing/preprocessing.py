from preprocessing.data_entry import Topic, DataEntry
from config import Config

from typing import List, Dict, Any
import random
from random import seed
import pandas as pd
from pathlib import Path


cfg = Config.get()


def preprocess_image_vision(image_vision: Dict[Any, Any]) -> Dict[str, float]:
    """
    Extract dictionary with strings of all image-vision outputs and corresponding scores
    :param image_vision: Dictionary with image_vision outputs for specific webpage
    :return: Dictionary with label_annotations and scores
    """
    label_annotations = dict()
    for label_annotation in image_vision["labelAnnotations"]:
        label_annotations.setdefault(label_annotation["description"], round(label_annotation["score"], 4))

    return label_annotations


def create_dataset(size_dataset: int = -1, set_seed: bool = True, topic_ids: List[int] = None):
    """
    Create DataFrame with image-ids, topic-ids and List with image-vision outputs; save dataset.pkl in working/
    :param size_dataset: Specify amount of images per topic (size_dataset > 0)
    :param set_seed: If True: seed(1) is used
    :param topic_ids: Specify topic-ids that should be used [51, 100]
    """
    # If no topic-ids are specified: Select all available topic-ids
    if topic_ids is None:
        topic_ids = [topic.number for topic in Topic.load_all()]

    # Set seed if set_seed = True
    if set_seed:
        seed(1)

    # Create a dataset for specific topics with image IDs of a specific sample size
    topic_images = dict()
    for topic_id in topic_ids:
        topic = Topic.get(topic_number=topic_id)
        image_ids = Topic.get_image_ids(topic)
        if size_dataset != -1:
            image_ids = random.choices(image_ids, k=size_dataset)
        topic_images.setdefault(topic_id, image_ids)

    # Get dataset with List of image-vision outputs
    dataset = pd.DataFrame
    first_iteration = True
    for topic_id in topic_images:
        for image_id in topic_images[topic_id]:
            image_vision = DataEntry.load(image_id=image_id).image_vision
            image_vision = preprocess_image_vision(image_vision=image_vision)
            data = {"image_id": image_id, "topic_id": topic_id, "data": image_vision}

            if first_iteration:
                dataset = pd.DataFrame(data=[data])
                first_iteration = False
            else:
                dataset = pd.concat([dataset, pd.DataFrame([data])], ignore_index=True)

    # Save DataFrame as pickle-file
    dataset.to_pickle(cfg.working_dir.joinpath(Path('dataset.pkl')))


def load_dataset() -> pd.DataFrame:
    """
    Load dataset
    :return: DataFrame with saved data
    """
    dataset = pd.read_pickle(cfg.working_dir.joinpath(Path('dataset.pkl')))

    return dataset
