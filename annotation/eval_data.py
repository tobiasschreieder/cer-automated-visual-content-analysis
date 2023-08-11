"""
The methods in eval_data.py follow, with slight variations, the work of:

Braker, J., Heinemann, L., Schreieder, T.: Aramis at touché 2022: Argument detection in pictures using machine learning.
Working Notes Papers of the CLEF (2022)

Carnot, M.L., Heinemann, L., Braker, J., Schreieder, T., Kiesel, J., Fröbe,
M., Potthast, M., Stein, B.: On Stance Detection in Image Retrieval for Argumentation. In: Proc. of SIGIR. ACM (2023)

Schreieder, T., Braker, J.: Touché 2022 Best of Labs: Neural Image Retrieval for Argumentation. CLEF (2023)

The original source code can be found at: https://github.com/webis-de/SIGIR-23
"""

import logging
from pathlib import Path
import pandas as pd

from config import Config


cfg = Config.get()
log = logging.getLogger('Evaluation')


eval_file = cfg.working_dir.joinpath(Path('image_eval.txt'))
if eval_file.exists():
    df = pd.read_csv(eval_file, sep=' ')
else:
    df = pd.DataFrame(columns=['image_id', 'user', 'Topic', 'Topic_correct'])

df.astype(dtype={
    'image_id': pd.StringDtype(),
    'user': pd.StringDtype(),
    'Topic': int,
    'Topic_correct': bool,
})
df.set_index(['image_id', 'user', 'Topic'], inplace=True)


def get_df() -> pd.DataFrame:
    return df.copy()


def save_df():
    df.to_csv(eval_file, sep=' ')
