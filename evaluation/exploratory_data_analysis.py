from preprocessing.preprocessing import load_dataset
from preprocessing.data_entry import Topic
from config import Config

from statistics import mean
from typing import Dict, Union
from pathlib import Path


cfg = Config.get()


def exploratory_data_analysis() -> Dict[int, Dict[str, Union[float, str]]]:
    """
    Create dictionary with exploratory-data-analysis statistics
    Calculate: Average Amount Elements per Image, Minimum Amount Elements per Image, Maximum Amount Elements per Image,
    Amount Empty Sets, Amount Unique Words
    :return: Dictionary with results per topic
    """
    dataset = load_dataset()
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

        average_amount_elements = mean(average_amount_elements)
        amount_unique_words = len(unique_words)

        topic_title = Topic.get(topic_number=topic_id).title
        stats = {"title": topic_title, "average_amount_elements": average_amount_elements,
                 "amount_empty_sets": amount_empty_sets, "min_amount_elements": min_amount_elements,
                 "max_amount_elements": max_amount_elements, "amount_unique_words": amount_unique_words}

        eda.setdefault(topic_id, stats)

    return eda


def create_eda_md_table(eda: Dict[int, Dict[str, Union[float, str]]]) -> str:
    """
    Create text for MD-File of exploratory data analysis
    :param eda: Dictionary with exploratory data analysis
    :return: String with MD-Table
    """
    text = "# Exploratory Data Analysis \n"
    text += ("| Topic-ID | Title | Average Amount Elements per Image | Minimum Amount Elements per Image | "
             "Maximum Amount Elements per Image| Amount Empty Sets | Amount Unique Words | \n")
    text += "|---|---|---|---|---|---|---| \n"
    for topic, stats in eda.items():
        text += ("| " + str(topic) + " | " + str(stats["title"]) + " | " + str(stats["average_amount_elements"]) + " | "
                 + str(stats["min_amount_elements"]) + " | " + str(stats["max_amount_elements"]) + " | "
                 + str(stats["amount_empty_sets"]) + " | " + str(stats["amount_unique_words"]) + " | \n")

    return text


def run_exploratory_data_analysis():
    """
    Run exploratory data analysis and save analysis as MD-File
    """
    eda = exploratory_data_analysis()
    text = [create_eda_md_table(eda=eda)]

    # Save eda as MD-File in output_dir
    with open(cfg.output_dir.joinpath(Path('exploratory_data_analysis.md')), 'w') as f:
        for item in text:
            f.write("%s\n" % item)
