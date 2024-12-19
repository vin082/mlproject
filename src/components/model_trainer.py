import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            
            # Convert to numpy arrays if not already in that format
            train_array = np.array(train_array)
            test_array = np.array(test_array)
            
            # Validate array dimensions
            if train_array.ndim != 2 or test_array.ndim != 2:
                raise CustomException("Train and test arrays must be 2-dimensional.", sys)
            
            logging.info(f"Train array shape: {train_array.shape}")
            logging.info(f"Test array shape: {test_array.shape}")
            
            # Split features and target variables
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            # Define models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # Use the `evaluate_models` utility function
            logging.info("Training and evaluating models using `evaluate_models`")
            model_scores = evaluate_models(X_train, y_train, X_test, y_test, models)
            
            # Log scores
            for model_name, score in model_scores.items():
                logging.info(f"{model_name} R² Score: {score}")

            # Select the best model
            best_model_name = max(model_scores, key=model_scores.get)
            best_score = model_scores[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with R² Score: {best_score}")

            # Validate best model performance
            if best_score < 0.6:
                raise CustomException("No model achieved a minimum R² score of 0.6", sys)

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Best model saved at: {self.model_trainer_config.trained_model_file_path}")

            return best_score

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise CustomException(e, sys)
