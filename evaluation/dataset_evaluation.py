'''
We evaluated the dataset by manually labeling 100 samples of each used topic.
This script creates two DataFrames, which are both stored as .csv files:
- eval_intercoder_reliability.csv, which contains the IR scores of the annotators
- eval_validity.csv, which contains the overall numbers regarding the belonging of images to these topics
'''

import pandas as pd
from config import Config

cfg = Config.get()

def run_dataset_evaluation(file_1, file_2):
    work_path = cfg.working_dir
    out_path = cfg.output_dir

    # load evaluation files
    labels_t = pd.read_csv(work_path.joinpath(file_1), sep=' ')
    labels_p = pd.read_csv(work_path.joinpath(file_2), sep=' ')

    labels_t.rename(columns={'Topic_correct': 'topic_correct_t'}, inplace=True)
    labels_p.rename(columns={'Topic_correct': 'topic_correct_p'}, inplace=True)

    # merge files, eliminate unwanted records
    labels = pd.merge(labels_t, labels_p, how='outer', on=['image_id']).dropna() \
        .drop_duplicates(subset='image_id', keep="first")
    
    # ensure typing
    labels['topic_correct_p'] = labels['topic_correct_p'].astype(bool)
    labels['topic_correct_t'] = labels['topic_correct_t'].astype(bool)

    # compute agreement between the annotators
    labels['agree'] = (labels['topic_correct_t'] == labels['topic_correct_p'])

    # create intercoder reliability DataFrame
    reliability = labels[['Topic_x', 'agree']].groupby(by='Topic_x').sum()
    sizes = labels[['Topic_x']].groupby(by='Topic_x').size()
    reliability['size'] = sizes
    reliability['intercoder_reliability'] = (reliability['agree'] / reliability['size'])

    reliability = reliability.reset_index().rename(columns={'Topic_x': 'topic_id'})

    # compute overall values
    agree_sum = reliability['agree'].sum()
    size_sum = reliability['size'].sum()
    ir_all = reliability['agree'].sum() / reliability['size'].sum()
    reliability.loc[len(reliability)] = ['all', agree_sum, size_sum, ir_all]

    # create topic validity DataFrame
    # merge annotators evaluations, clean DataFrame
    validity_p = labels[['Topic_y', 'topic_correct_p']]
    validity_t = labels[['Topic_x', 'topic_correct_t']]
    validity = pd.merge(validity_p, validity_t, left_index=True, right_index=True) \
        .drop(columns='Topic_y').rename(columns={'Topic_x': 'topic'})
    
    # add column for cases where both annotated true
    validity['topic_correct_both'] = validity['topic_correct_p'] & validity['topic_correct_t']

    # sum up by topic and create new columns
    validity = validity.groupby(by='topic').sum()
    validity['size'] = sizes
    validity['percent_correct_p'] = (validity['topic_correct_p'] / validity['size'])
    validity['percent_correct_t'] = (validity['topic_correct_t'] / validity['size'])
    validity['percent_correct_both'] = (validity['topic_correct_both'] / validity['size'])

    validity = validity.reset_index()

    # save both DataFrames
    reliability.to_csv(out_path.joinpath('eval_intercoder_reliability.csv'))
    validity.to_csv(out_path.joinpath('eval_topic_validity.csv'))


def print_dataset_evaluation():
    out_path = cfg.output_dir
    reliability = pd.read_csv(out_path.joinpath('eval_intercoder_reliability.csv'), index_col=0)
    validity = pd.read_csv(out_path.joinpath('eval_topic_validity.csv'), index_col=0)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('Evaluation of intercoder reliability:')
        print(reliability)
        print('\nEvaluation of topic validity:')
        print(validity)
