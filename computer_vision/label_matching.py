from preprocessing.data_entry import Topic
from preprocessing.preprocessing import load_dataset
from config import Config

import pandas as pd
from Levenshtein import ratio
from typing import List, Dict
from pathlib import Path


cfg = Config.get()


def exact_match(dataset_touche: pd.DataFrame, dataset_clarifai: pd.DataFrame) -> Dict[int, Dict[str, int]]:
    """
    Calculate exact matches for two given datasets
    :param dataset_touche: Dataframe with Touché dataset
    :param dataset_clarifai: Dataframe with Clarifai dataset
    :return: Dictionary with results
    """
    results = dict()
    topic_ids = list(set(dataset_touche["topic_id"].tolist()))
    for topic_id in topic_ids:
        data_touche = dataset_touche[dataset_touche["topic_id"] == topic_id]
        data_clarifai = dataset_clarifai[dataset_clarifai["topic_id"] == topic_id]

        elements_touche = list()
        for element in data_touche["data"]:
            elements_touche.append(element)

        elements_clarifai = list()
        for element in data_clarifai["data"]:
            elements_clarifai.append(element)

        unique_words_touche = list()
        for element in elements_touche:
            for word in element:
                if word not in unique_words_touche:
                    unique_words_touche.append(word)

        unique_words_clarifai = list()
        for element in elements_clarifai:
            for word in element:
                if word not in unique_words_clarifai:
                    unique_words_clarifai.append(word)

        matches = list()
        for word in unique_words_touche:
            if word in unique_words_clarifai:
                matches.append(word)

        topic_results = {"matches": len(matches), "touche": len(unique_words_touche),
                         "clarifai": len(unique_words_clarifai)}

        results.setdefault(topic_id, topic_results)

    return results


def levenshtein_match(dataset_touche: pd.DataFrame, dataset_clarifai: pd.DataFrame, threshold: float = 0.8) \
        -> Dict[int, Dict[str, int]]:
    """
    Calculate Levenshtein matches for two given datasets (Match if similarity >= threshold)
    :param dataset_touche: Dataframe with Touché dataset
    :param dataset_clarifai: Dataframe with Clarifai dataset
    :param threshold: Specify threshold for Levenshtein similarity
    :return: Dictionary with results
    """
    results = dict()
    topic_ids = list(set(dataset_touche["topic_id"].tolist()))
    for topic_id in topic_ids:
        data_touche = dataset_touche[dataset_touche["topic_id"] == topic_id]
        data_clarifai = dataset_clarifai[dataset_clarifai["topic_id"] == topic_id]

        elements_touche = list()
        for element in data_touche["data"]:
            elements_touche.append(element)

        elements_clarifai = list()
        for element in data_clarifai["data"]:
            elements_clarifai.append(element)

        unique_words_touche = list()
        for element in elements_touche:
            for word in element:
                if word not in unique_words_touche:
                    unique_words_touche.append(word)

        unique_words_clarifai = list()
        for element in elements_clarifai:
            for word in element:
                if word not in unique_words_clarifai:
                    unique_words_clarifai.append(word)

        matches = list()
        for word_t in unique_words_touche:
            for word_c in unique_words_clarifai:
                similarity = ratio(s1=word_t, s2=word_c)
                if similarity >= threshold:
                    matches.append(word_t)
                    break

        topic_results = {"matches": len(matches), "touche": len(unique_words_touche),
                         "clarifai": len(unique_words_clarifai)}

        results.setdefault(topic_id, topic_results)

    return results


def create_eda_md_table(results_exact_matches: Dict[int, Dict[str, int]],
                        results_levenshtein_matches: Dict[int, Dict[str, int]]) -> str:
    """
    Create text for MD-File of exploratory data analysis
    :param results_exact_matches: Dictionary with match results from exact matches
    :param results_levenshtein_matches: Dictionary with match results from levenshtein matches
    :return: String with MD-Table
    """
    text = "# Touche-Clarifai Matching Results \n"
    text += ("| Topic-ID | Title | Amount Unique Labels Touche | Amount Unique Labels Clarifai | Exact Matches | "
             "Levenshtein Matches | \n")
    text += "|---|---|---|---|---|---| \n"
    for topic_id in results_exact_matches:
        text += ("| " + str(topic_id) + " | " + str(Topic.get(topic_number=topic_id).title) + " | " +
                 str(results_exact_matches[topic_id]["touche"]) + " | " +
                 str(results_exact_matches[topic_id]["clarifai"]) + " | " +
                 str(results_exact_matches[topic_id]["matches"]) + " | " +
                 str(results_levenshtein_matches[topic_id]["matches"]) + " | \n")

    return text


def run_label_matching(size_dataset: int = -1, threshold: float = 0.8, topic_ids: List[int] = None):
    """
    Run label matching with exact matches and levenshtein matches
    :param size_dataset: Specify dataset size
    :param threshold: Specify threshold for levenshtein similarity matching
    :param topic_ids: Specify topic-ids
    """
    dataset_touche = load_dataset(size_dataset=size_dataset, topic_ids=topic_ids)
    dataset_clarifai = load_dataset(size_dataset=size_dataset, topic_ids=topic_ids, use_clarifai_data=True)

    results_exact_matches = exact_match(dataset_touche=dataset_touche, dataset_clarifai=dataset_clarifai)
    results_levenshtein_matches = levenshtein_match(dataset_touche=dataset_touche, dataset_clarifai=dataset_clarifai,
                                                    threshold=threshold)

    text = [create_eda_md_table(results_exact_matches=results_exact_matches,
                                results_levenshtein_matches=results_levenshtein_matches)]

    with open(cfg.output_dir.joinpath(Path("label_matches_touche_clarifai.md")), 'w') as f:
        for item in text:
            f.write("%s\n" % item)

    print("Label matching saved at: " + str(cfg.output_dir.joinpath(Path("label_matches_touche_clarifai.md"))) + ".")
