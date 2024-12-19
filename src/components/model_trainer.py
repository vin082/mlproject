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
from sklearn.metrics import r2_score
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
            train_array = np.array(train_array)
            test_array = np.array(test_array)
            logging.info(f"Train array shape: {train_array.shape}")
            logging.info(f"Test array shape: {test_array.shape}")

            
            # Split the data
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]
             # Check if the arrays are 2D
            if train_array.ndim != 2:
              raise CustomException("train_array is not a 2D array")
            if test_array.ndim != 2:
              raise CustomException("test_array is not a 2D array")
            # Add more debug prints
            logging.info(f"X_train shape: {X_train.shape}")
            logging.info(f"y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}")
            logging.info(f"y_test shape: {y_test.shape}")
            # Define models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            logging.info("Training and evaluating models")
            
            # Train and evaluate each model
            best_model = None
            best_score = float('-inf')
            best_model_name = None

            for model_name, model in models.items():
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate score
                score = r2_score(y_test, y_pred)
                
                logging.info(f"{model_name} R2 Score: {score}")
                
                # Update best model if current score is better
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = model_name

            logging.info(f"Best performing model: {best_model_name}")
            logging.info(f"Best score: {best_score}")

            if best_score < 0.6:
                raise CustomException("No model achieved minimum R² score of 0.6")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            logging.info(f"Best model saved at: {self.model_trainer_config.trained_model_file_path}")

            # Final predictions using best model
            predicted =  best_model.predict(X_test)
            final_score = r2_score(y_test, predicted)
            
            return final_score

        except Exception as e:
            logging.error("Error in model training")
            raise CustomException(e, sys)