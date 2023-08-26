'''
We evaluated the dataset by manually labeling 100 samples of each used topic.
This script creates two DataFrames, which are both stored as .csv files:
- eval_intercoder_reliability.csv, which contains the IR scores of the annotators
- eval_validity.csv, which contains the overall numbers regarding the belonging of images to these topics
'''

import pandas as pd
from config import Config
import krippendorff

cfg = Config.get()

def run_dataset_evaluation(file_1, file_2):
    work_path = cfg.working_dir
    out_path = cfg.output_dir

    # load evaluation files
    labels_a = pd.read_csv(work_path.joinpath(file_1), sep=' ')
    labels_b = pd.read_csv(work_path.joinpath(file_2), sep=' ')

    labels_a.rename(columns={'Topic_correct': 'topic_correct_a'}, inplace=True)
    labels_b.rename(columns={'Topic_correct': 'topic_correct_b'}, inplace=True)

    # merge files, eliminate unwanted records
    labels = pd.merge(labels_a, labels_b, how='outer', on=['image_id']).dropna() \
        .drop_duplicates(subset='image_id', keep="first")
    
    # ensure typing
    labels['topic_correct_b'] = labels['topic_correct_b'].astype(bool)
    labels['topic_correct_a'] = labels['topic_correct_a'].astype(bool)

    # compute agreement between the annotators
    labels['agree'] = (labels['topic_correct_a'] == labels['topic_correct_b'])

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

    # compute Krippendorff's Alpha for each topic
    alphas = []
    for topic in labels['Topic_x'].unique():
        reliability_data = labels.loc[labels['Topic_x'] == topic, ['topic_correct_b', 'topic_correct_a']].astype(int)
        ka = krippendorff.alpha(reliability_data.transpose(), level_of_measurement='nominal')
        alphas.append(ka)

    # compute the overall alpha
    alphas.append(krippendorff.alpha(labels[['topic_correct_b', 'topic_correct_a']].astype(int).transpose(), level_of_measurement='nominal'))

    # add to reliability dataframe
    reliability['krippendorff'] = alphas

    # create topic validity DataFrame
    # merge annotators evaluations, clean DataFrame
    validity_b = labels[['Topic_y', 'topic_correct_b']]
    validity_a = labels[['Topic_x', 'topic_correct_a']]
    validity = pd.merge(validity_b, validity_a, left_index=True, right_index=True) \
        .drop(columns='Topic_y').rename(columns={'Topic_x': 'topic'})
    
    # add column for cases where both annotated true
    validity['topic_correct_both'] = validity['topic_correct_b'] & validity['topic_correct_a']

    # sum up by topic and create new columns
    validity = validity.groupby(by='topic').sum()
    validity['size'] = sizes
    validity['percent_correct_b'] = (validity['topic_correct_b'] / validity['size'])
    validity['percent_correct_a'] = (validity['topic_correct_a'] / validity['size'])
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
