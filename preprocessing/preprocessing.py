import operator

from preprocessing.data_entry import Topic, DataEntry, Clarifai
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
        label_annotations.setdefault(label_annotation["description"].lower(), round(label_annotation["score"], 4))

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
        if size_dataset != -1 and len(image_ids) > size_dataset:
            image_ids = random.choices(image_ids, k=size_dataset)
        topic_images.setdefault(topic_id, image_ids)

    # Get dataset with List of image-vision outputs
    dataset = pd.DataFrame
    first_iteration = True
    for topic_id in topic_images:
        for image_id in topic_images[topic_id]:
            image_vision = DataEntry.load(image_id=image_id).image_vision
            image_vision = preprocess_image_vision(image_vision=image_vision)

            # Create DataFrame
            data = {"image_id": image_id, "topic_id": topic_id, "data": image_vision}
            if first_iteration:
                dataset = pd.DataFrame(data=[data])
                first_iteration = False
            else:
                dataset = pd.concat([dataset, pd.DataFrame([data])], ignore_index=True)

    # Save DataFrame as pickle-file
    dataset.to_pickle(cfg.working_dir.joinpath(Path('dataset.pkl')))
    print("Dataset saved at working/dataset.pkl.")


def load_dataset() -> pd.DataFrame:
    """
    Load dataset
    :return: DataFrame with saved data
    """
    dataset = pd.read_pickle(cfg.working_dir.joinpath(Path('dataset.pkl')))

    return dataset


def create_clarifai_dataset(size_dataset: int = -1, set_seed: bool = True, topic_ids: List[int] = None,
                            max_amount_labels: int = 10):
    """
    Create Clarifai DataFrame with image-ids, topic-ids and List with image-vision outputs; save clarifai_dataset.pkl
    in working/
    :param size_dataset: Specify amount of images per topic (size_dataset > 0)
    :param set_seed: If True: seed(1) is used
    :param topic_ids: Specify topic-ids that should be used [51, 100]
    :param max_amount_labels: Specify maximum amount of extracted labels in decreasing score order
    """
    # If no topic-ids are specified: Select all available topic-ids
    if topic_ids is None:
        topic_ids = [topic.number for topic in Topic.load_all()]

    # Set seed if set_seed = True
    if set_seed:
        seed(1)

    # Create a dataset for specific topics with image IDs
    topic_images = dict()
    for topic_id in topic_ids:
        topic = Topic.get(topic_number=topic_id)
        image_ids = Topic.get_image_ids(topic)
        topic_images.setdefault(topic_id, image_ids)

    # Create Clarifai dataset
    clarifai_image_ids = Clarifai.get_image_ids()
    clarifai_topic_images = dict()
    for topic_id, image_ids in topic_images.items():
        correct_image_ids = list()
        for image_id in image_ids:
            if image_id in clarifai_image_ids:
                correct_image_ids.append(image_id)

        # Choose sample of image_ids
        if size_dataset != -1 and len(correct_image_ids) > size_dataset:
            correct_image_ids = random.choices(correct_image_ids, k=size_dataset)
        clarifai_topic_images.setdefault(topic_id, correct_image_ids)

    # Get dataset with List of image-vision outputs
    dataset = pd.DataFrame
    first_iteration = True
    for topic_id in clarifai_topic_images:
        for image_id in clarifai_topic_images[topic_id]:
            image_vision_original = Clarifai.load(image_id=image_id).image_vision
            image_vision_original = dict(sorted(image_vision_original.items(), key=operator.itemgetter(1),
                                                reverse=True))

            # Round label values and test max amount of labels
            image_vision = dict()
            count_labels = 0
            for label, score in image_vision_original.items():
                if max_amount_labels is not None:
                    if count_labels < max_amount_labels:
                        image_vision.setdefault(label, round(score, 4))
                        count_labels += 1
                else:
                    image_vision.setdefault(label, round(score, 4))

            # Create DataFrame
            data = {"image_id": image_id, "topic_id": topic_id, "data": image_vision}
            if first_iteration:
                dataset = pd.DataFrame(data=[data])
                first_iteration = False
            else:
                dataset = pd.concat([dataset, pd.DataFrame([data])], ignore_index=True)

    # Save DataFrame as pickle-file
    dataset.to_pickle(cfg.working_dir.joinpath(Path('clarifai_dataset.pkl')))
    print("Clarifai dataset saved at working/clarifai_dataset.pkl.")


def load_clarifai_dataset() -> pd.DataFrame:
    """
    Load Clarifai dataset
    :return: DataFrame with saved data
    """
    dataset = pd.read_pickle(cfg.working_dir.joinpath(Path('clarifai_dataset.pkl')))

    return dataset
