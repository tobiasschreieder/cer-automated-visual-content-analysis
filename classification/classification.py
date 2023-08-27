from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from time import time
import json

from config import Config


cfg = Config.get()


class Classifier:
    def __init__(self, clf, data:pd.DataFrame, topic_id: int, grid_params: list = None, train_size: float = 0.9,
                 use_likelihood: bool = False, seed: int = 1, dataset_name: str = 'not specified'):
        """
        Creates an instance of the Classifier.
        :param clf: Instance of the classifier to be used.
        :param data: Data set to use for classification.
        :param topic_id: Topic to use as predicted variable.
        :param train_size: Specification of the train-test-split.
        :param use_likelihood: indicates whether likelihood or binary coding of the data is to be used.
        :param seed: Seeding value for the train-test-split.
        :param dataset_name: Name of the dataset to be stored in a JSON evaluation file.
        """
        # general parameters
        self.use_likelihood = use_likelihood
        self.seed = seed
        self.pred_topic_id = topic_id

        # classifier, its name and its initial parameters
        self.clf = clf
        self.model_name = str(clf).replace('()', '')
        self.model_params = clf.get_params()

        # dataset
        self.dataset_name = dataset_name
        self.X_train, self.X_test, self.y_train, self.y_test = (
            self._create_model_input(data, topic_id, train_size=train_size, use_likelihood=use_likelihood,
                                     random_state=seed))

        # grid search regarding
        self.grid_params = grid_params
        self.best_score = 0

    def fit(self):
        """
        Fits the model of this instance to its data.
        """
        self.clf.fit(self.X_train, self.y_train)

    def predict(self, X = None):
        """
        Predicts data with the model of this instance. If no data is specified, test data of this instance will be used.
        :param X: dataset to predict from
        :return: a vector y with the predictions for X.
        """
        if X is None:
            return self.clf.predict(self.X_test)
        return self.clf.predict(X)

    def evaluate(self, y_pred = None, output=True, save=True):
        """
        Evaluates the classifier of the instance on a specified predicted set of data.
        If no test set is specified, the set will be computed through prediction of self.X_test.
        :param y_pred: Predicted values to evaluate
        :param output: Output results to console
        :param save: Save output as a json-file
        """
        # evaluate on predictions of test set
        if y_pred is None:
            y_pred = self.predict(self.X_test)

        # compute metrics
        acc = accuracy_score(self.y_test, y_pred)
        pre = precision_score(self.y_test, y_pred)
        rec = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        f1_weighted = f1_score(self.y_test, y_pred, average='weighted')

        # optional: print results
        if output:
            print(f'Model type: {self.model_name}')
            print(f'Model parameters: {self.model_params}')
            print(f'accuracy:  {acc}\nprecision: {pre}\nrecall:    {rec}\nf1_weighted:  {f1}\n')

        # optional: save to json-file
        if save:
            file_name = f'eval_{self.model_name}_{time()}.json'
            classifier_path = cfg.output_dir.joinpath("classifier")
            Path(classifier_path).mkdir(parents=True, exist_ok=True)
            doc = {
                'dataset': self.dataset_name,
                'use_likelihood': self.use_likelihood,
                'pred_topic_id': self.pred_topic_id,
                'model': self.model_name,
                'model_params': self.model_params,
                'best_score_f1': self.best_score,
                'eval_metrics': {
                    'accuracy': acc,
                    'precision': pre,
                    'recall': rec,
                    'f1': f1,
                    'f1_weighted': f1_weighted
                },
            }
            with open(classifier_path.joinpath(Path(file_name)), 'w') as f:
                json.dump(doc, f)

    def evaluate_grid_search(self):
        """
        Performs a grid search on the specified grid parameters of this instance.
        Calls the self.evaluate method for further analyzing the best model of the search. 
        """
        # defining and starting the grid search
        classifier = GridSearchCV(estimator=self.clf, param_grid=self.grid_params, scoring='f1', refit=True, verbose=2)
        classifier.fit(self.X_train, self.y_train)

        # store the best classifier, its parameters and score
        self.clf = classifier.best_estimator_
        self.model_params = classifier.best_params_
        self.best_score = classifier.best_score_
        
        # evaluate the best classifier
        self.evaluate()

    def _create_model_input(self, df_source: pd.DataFrame, topic_id: int, train_size: float = 0.9,
                            use_likelihood: bool = False, random_state: int = 1) -> Any:
        """
        Reshape data to input format for ML-models and perform a train-test-split
        :param df_source: DataFrame to use
        :param topic_id: ID of topic that represents the dependent variable
        :param use_likelihood: determine wether to use binary or likelihood values
        :return: Matrix X_train and X_test of shape (n_images / , n_features), array y with length n_images
        """
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
        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=train_size,
                                                            random_state=random_state)

        return x_train, x_test, y_train, y_test
    