import os
import sys
import numpy as np
import pandas as pd
import pickle
from src.logger import logging
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models,params):
    try:
        report = {}

        #for i in range(len(models)):
            #model = list(models.values())[i]
        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")
            # Handle CatBoost or any other non-Sklearn compliant models
            if model_name in["CatBoost Regressor","XGBRegressor"]:
                logging.info(f"Fitting {model_name} without GridSearchCV due to compatibility issues")
                model.fit(X_train, y_train, verbose=False)
                y_test_pred = model.predict(X_test)
                test_model_score = r2_score(y_test, y_test_pred)
                report[model_name] = test_model_score
                continue
            #list all the params of the model
            param_grid= params.get(model_name, {})
            if param_grid:
                logging.info(f"Tuning hyperparameters for {model_name}")
                grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2')
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")

            else:
                logging.info(f"No hyperparameters provided for {model_name}, using default configuration")
                model.fit(X_train, y_train)
                best_model = model

            # Train model
            #model.fit(X_train, y_train)

            #model.fit(X_train, y_train)

            #y_train_pred = model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            #train_model_score = r2_score(y_train, y_train_pred)
            test_model_score =  r2_score(y_test, y_test_pred)

            #report[list(models.keys())[i]] = test_model_score
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)