"""
The methods in evaluation.py follow, with slight variations, the work of:

Braker, J., Heinemann, L., Schreieder, T.: Aramis at touché 2022: Argument detection in pictures using machine learning.
Working Notes Papers of the CLEF (2022)

Carnot, M.L., Heinemann, L., Braker, J., Schreieder, T., Kiesel, J., Fröbe,
M., Potthast, M., Stein, B.: On Stance Detection in Image Retrieval for Argumentation. In: Proc. of SIGIR. ACM (2023)

Schreieder, T., Braker, J.: Touché 2022 Best of Labs: Neural Image Retrieval for Argumentation. CLEF (2023)

The original source code can be found at: https://github.com/webis-de/SIGIR-23
"""

import logging
from typing import Tuple, Dict

from config import Config
from preprocessing.data_entry import Topic, DataEntry
from annotation.eval_data import df, save_df


cfg = Config.get()
log = logging.getLogger('Evaluation')


def has_eval(image_id: str, topic: int = None) -> bool:
    if topic:
        try:
            return len(df.loc[(image_id, slice(None), topic), :]) > 0
        except KeyError:
            return False
    return image_id in df.index.get_level_values(0)


def get_image_to_eval(topic: Topic) -> DataEntry or None:
    for image in topic.get_image_ids():
        if has_eval(image, topic.number):
            continue
        return DataEntry.load(image)
    return None


def get_eval(image_id: str, topic: int) -> Tuple[int] or None:
    if has_eval(image_id):
        temp = df.loc[(image_id, slice(None), topic), :]
        return temp.loc[temp.index[0], 'Topic']
    return None


def get_evaluations(image_id: str, topic: int) -> Dict[str, Tuple[int]] or None:
    if has_eval(image_id, topic):
        temp = df.loc[(image_id, slice(None), topic), :]
        evals = []
        for user in temp.index:
            evals.append((temp.loc[user, 'Topic']))
        return evals
    return None


def save_eval(image_id: str, user: str, topic: int, topic_correct: bool) -> None:
    df.loc[(image_id, user, topic), :] = topic_correct
    save_df()
    log.debug('Saved evaluation for %s %s %s: %s %s %s', image_id, user, topic, topic_correct)
