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

    # Test working/datasets path
    path = cfg.working_dir.joinpath("datasets")
    Path(path).mkdir(parents=True, exist_ok=True)

    # Create a dataset for specific topics with image IDs of a specific sample size
    topic_images = dict()
    for topic_id in topic_ids:
        topic = Topic.get(topic_number=topic_id)
        image_ids = Topic.get_image_ids(topic)
        if size_dataset != -1 and len(image_ids) > size_dataset:
            image_ids = random.sample(image_ids, k=size_dataset)
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
    if size_dataset == -1:
        size_dataset = "max"

    filename = "dataset_touche_topics="
    for topic_id in topic_ids:
        filename += str(topic_id) + "+"
    filename = filename[:-1] + "_size=" + str(size_dataset) + ".pkl"

    dataset.to_pickle(path.joinpath(Path(filename)))
    print("Dataset saved at working/dataset.pkl.")


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

    # Test working/datasets path
    datasets_path = cfg.working_dir.joinpath("datasets")
    Path(datasets_path).mkdir(parents=True, exist_ok=True)

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
            correct_image_ids = random.sample(correct_image_ids, k=size_dataset)
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
    if size_dataset == -1:
        size_dataset = "max"

    filename = "dataset_clarifai_topics="
    for topic_id in topic_ids:
        filename += str(topic_id) + "+"
    filename = filename[:-1] + "_size=" + str(size_dataset) + ".pkl"

    dataset.to_pickle(datasets_path.joinpath(Path(filename)))
    print("Clarifai dataset saved at working/datasets/" + filename + ".")


def load_dataset(size_dataset: int = -1, use_clarifai_data: bool = False,
                 topic_ids: List[int] = None) -> pd.DataFrame:
    """
    Load dataset (Touché, Clarifai)
    :param size_dataset: Specify amount of images per topic (size_dataset > 0)
    :param use_clarifai_data: If True the Clarifai dataset will be returned instead of the Touché dataset
    :param topic_ids: Specify topic-ids that should be used [51, 100]
    :return: DataFrame with saved data
    """
    path = cfg.working_dir.joinpath(Path("datasets"))
    filename = str()
    dataset = pd.DataFrame

    # Load Touché or Clarifai dataset
    # Touché or Clarifai
    if use_clarifai_data:
        filename += "dataset_clarifai_topics="
    else:
        filename += "dataset_touche_topics="

    # Topic-ids
    if topic_ids is None:
        topic_ids = [topic.number for topic in Topic.load_all()]
    for topic_id in topic_ids:
        filename += str(topic_id) + "+"
    filename = filename[:-1] + "_size="

    # Size dataset
    if size_dataset > 0:
        filename += str(size_dataset)
    else:
        filename += "max"
    filename += ".pkl"

    # Load dataset
    try:
        dataset = pd.read_pickle(path.joinpath(Path(filename)))
    except FileNotFoundError:
        print("FileNotFoundError: Dataset with this specifications does not exist in working!")
        print("Trying to create dataset..")
        if use_clarifai_data:
            create_clarifai_dataset(size_dataset=size_dataset, topic_ids=topic_ids)
        else:
            create_dataset(size_dataset=size_dataset, topic_ids=topic_ids)
        load_dataset(size_dataset=size_dataset, topic_ids=topic_ids, use_clarifai_data=use_clarifai_data)

    return dataset
