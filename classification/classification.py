from sklearn.metrics import *
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV

from time import time
import json

from config import Config

cfg = Config.get()

class Classifier():

    # TODO: load dataset directly here - just take the filename as an input!
    def __init__(self, clf, data:pd.DataFrame, topic_id:int, grid_params:list=None, train_size:float=0.9, use_likelihood:bool=False, seed:int=1, dataset_name:str='not specified'):
        self.clf = clf
        self.grid_params = grid_params
        self.model_params = clf.get_params()
        self.model_name = str(clf).replace('()', '')
        self.X_train, self.X_test, self.y_train, self.y_test = self._create_model_input(data, topic_id, train_size=train_size, use_likelihood=use_likelihood, random_state=seed)
        self.seed = seed
        self.topic_ids = self._get_topic_ids(data)
        self.dataset_name = dataset_name
        self.best_score = 0


    def fit(self):
        self.clf.fit(self.X_train, self.y_train)


    def predict(self, X=None):
        if X is None:
            return self.clf.predict(self.X_test)
        return self.clf.predict(X)


    def evaluate(self, y_pred=None, output=True, save=True):

        # evaluate on predictions of test set
        if y_pred is None:
            y_pred = self.predict(self.X_test)

        # metrics
        acc = accuracy_score(self.y_test, y_pred)
        pre = precision_score(self.y_test, y_pred)
        rec = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred) # f1micro/macro??

        if output:
            print(f'Model type: {self.model_name}')
            print(f'Model parameters: {self.model_params}')
            print(f'accuracy:  {acc}\nprecision: {pre}\nrecall:    {rec}\nf1_score:  {f1}\n')

        if save:
            file_name = f'eval_{self.model_name}_{time()}.json'
            classifier_path = cfg.working_dir.joinpath("classifier")
            Path(classifier_path).mkdir(parents=True, exist_ok=True)
            doc = {
                'dataset': self.dataset_name,
                'topics': self.topic_ids,
                'model': self.model_name,
                'model_params': self.model_params,
                'best_score': self.best_score,
                'eval_metrics': {
                    'accuracy': acc,
                    'precision': pre,
                    'recall': rec,
                    'f1': f1
                },
            }

            with open(classifier_path.joinpath(Path(file_name)), 'w') as f:
                json.dump(doc, f)


    def evaluate_grid_search(self):
        print(f'\nStarting Grid Search\n')
        classifier = GridSearchCV(estimator=self.clf, param_grid=self.grid_params, scoring='f1_weighted', refit=True, verbose=2)
        classifier.fit(self.X_train, self.y_train)
        self.clf = classifier.best_estimator_
        self.model_params = classifier.best_params_
        self.best_score = classifier.best_score_
        print(f'\nGrid Search done, best score: {self.best_score}\n')
        self.evaluate()


    def _create_model_input(self, df_source:pd.DataFrame, topic_id:int, train_size:float=0.9, use_likelihood:bool=False, random_state:int=1) -> pd.DataFrame:
        '''
        Reshape data to input format for ML-models and perform a train-test-split
        :param df_source: DataFrame to use
        :param topic_id: ID of topic that represents the dependent variable
        :param use_likelihood: determine wether to use binary or likelihood values
        :return: Matrix X_train and X_test of shape (n_images / , n_features), array y with length n_images
        '''
        # extract all features
        features = set({})
        for _, row in df_source.iterrows():
            features = features.union(set(row['data'].keys()))
        
        # create new dataframe with features as columns, dataframe for topic
        X = pd.DataFrame(index=df_source.index, columns=list(features)).fillna(0)
        y = np.zeros(X.shape[0])
        
        # fill dataframe
        for idx, row in df_source.iterrows():
            
            # set dependent variable
            if row['topic_id'] == topic_id:
                y[idx] = 1
                
            # set other variables
            keys = row['data'].keys()
            keys = [k for k in keys if k in features]
            for k in keys:
                if use_likelihood:
                    X.loc[idx, k] = row['data'][k]
                else:
                    X.loc[idx, k] = 1

        # train-test-split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)

        return X_train, X_test, y_train, y_test
    
    def _get_topic_ids(self, data:pd.DataFrame) -> list[str]:
        return data['topic_id'].unique().tolist()
    

