from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from catboost import CatBoostRegressor
from loguru import logger
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from app.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: Path = Path('artifacts/model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    @logger.catch
    def initiate_model_trainer(self, train_array, test_array):
        logger.info('Split training and test input data')
        X_train, y_train, X_test, y_test = (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1],
        )

        models = {
            'Random Forest': RandomForestRegressor(),
            'Decision Tree': DecisionTreeRegressor(),
            'Gradient Boost': GradientBoostingRegressor(),
            'Linear Regression': LinearRegression(),
            'K-Neighbors Regression': KNeighborsRegressor(),
            'XGBRegressor': XGBRegressor(),
            'CatBoosting Regressor': CatBoostRegressor(verbose=False),
            'AdaBoost Regressor': AdaBoostRegressor(),
        }

        model_report = evaluate_models(
            X_train, y_train, X_test, y_test, models
        )

        best_model_name, best_model_score = max(
            model_report.items(), key=lambda x: x[1]
        )

        if best_model_score < 0.6:
            raise Exception('No best model found!')

        logger.success(
            f'Best model found on both training and testing dataset, Name: "{best_model_name}" Score: {best_model_score:.2f}'
        )
        best_model = models[best_model_name]
        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model,
        )
        predicted = best_model.predict(X_test)
        r2_square = r2_score(y_test, predicted)

        return r2_square
