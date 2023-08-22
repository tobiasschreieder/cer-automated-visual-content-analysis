import pandas as pd

from preprocessing.preprocessing import load_dataset
from preprocessing.data_entry import Topic
from config import Config

from statistics import mean
from typing import Dict, Union, List
from pathlib import Path


cfg = Config.get()


def exploratory_data_analysis(dataset: pd.DataFrame) -> Dict[int, Dict[str, Union[float, str]]]:
    """
    Create dictionary with exploratory-data-analysis statistics
    Calculate: Average Amount Elements per Image, Minimum Amount Elements per Image, Maximum Amount Elements per Image,
    Amount Empty Sets, Amount Unique Words
    :param dataset: Dataframe with dataset to analyse
    :return: Dictionary with results per topic
    """
    topic_ids = list(set(dataset["topic_id"].tolist()))

    eda = dict()
    for topic_id in topic_ids:
        data = dataset[dataset["topic_id"] == topic_id]
        elements = list()
        for element in data["data"]:
            elements.append(element)

        average_amount_elements = list()
        unique_words = list()
        amount_empty_sets = 0
        min_amount_elements = 99999
        max_amount_elements = 0
        for element in elements:
            average_amount_elements.append(len(element))
            for word in element:
                if word not in unique_words:
                    unique_words.append(word)
            if len(element) == 0:
                amount_empty_sets += 1
            if len(element) > max_amount_elements:
                max_amount_elements = len(element)
            if len(element) < min_amount_elements:
                min_amount_elements = len(element)

        average_amount_elements = round(mean(average_amount_elements), 2)
        amount_unique_words = len(unique_words)
        amount_images = len(data)

        topic_title = Topic.get(topic_number=topic_id).title
        stats = {"title": topic_title, "amount_images": amount_images,
                 "average_amount_elements": average_amount_elements, "amount_empty_sets": amount_empty_sets,
                 "min_amount_elements": min_amount_elements, "max_amount_elements": max_amount_elements,
                 "amount_unique_words": amount_unique_words}

        eda.setdefault(topic_id, stats)

    return eda


def create_eda_md_table(eda: Dict[int, Dict[str, Union[float, str]]]) -> str:
    """
    Create text for MD-File of exploratory data analysis
    :param eda: Dictionary with exploratory data analysis
    :return: String with MD-Table
    """
    text = "# Exploratory Data Analysis \n"
    text += ("| Topic-ID | Title | Amount Images | Average Amount Labels per Image | "
             "Minimum Amount Labels per Image | Maximum Amount Labels per Image| Amount Empty Sets | "
             "Amount Unique Labels | \n")
    text += "|---|---|---|---|---|---|---|---| \n"
    for topic, stats in eda.items():
        text += ("| " + str(topic) + " | " + str(stats["title"]) + " | " + str(stats["amount_images"]) + " | "
                 + str(stats["average_amount_elements"]) + " | " + str(stats["min_amount_elements"]) + " | "
                 + str(stats["max_amount_elements"]) + " | " + str(stats["amount_empty_sets"]) + " | "
                 + str(stats["amount_unique_words"]) + " | \n")

    return text


def run_exploratory_data_analysis(size_dataset: int = -1, use_clarifai_data: bool = False, topic_ids: List[int] = None):
    """
    Run exploratory data analysis and save analysis as MD-File
    :param size_dataset: Specify amount of images per topic (size_dataset > 0)
    :param use_clarifai_data: If True the Clarifai dataset will be returned instead of the Touché dataset
    :param topic_ids: Specify topic-ids that should be used [51, 100]
    """
    dataset = load_dataset(size_dataset=size_dataset, use_clarifai_data=use_clarifai_data, topic_ids=topic_ids)
    eda = exploratory_data_analysis(dataset=dataset)
    text = [create_eda_md_table(eda=eda)]

    # Save eda as MD-File in output_dir
    if use_clarifai_data:
        filename = 'clarifai_exploratory_data_analysis.md'
    else:
        filename = 'touche_exploratory_data_analysis.md'

    with open(cfg.output_dir.joinpath(Path(filename)), 'w') as f:
        for item in text:
            f.write("%s\n" % item)
