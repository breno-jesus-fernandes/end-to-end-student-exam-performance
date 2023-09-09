from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: Path = Path('artitacts/preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    @logger.catch
    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        This method is responsible for data trasnformation
        """

        numerical_columns = ['writing_score', 'reading_score']
        categorical_columns = [
            'gender',
            'race_ethnicity',
            'parental_level_of_education',
            'lunch',
            'test_preparation_course',
        ]

        numerical_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False)),
            ]
        )

        logger.success('Numerical columns stander scaling completed')

        categorical_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False)),
            ]
        )

        logger.success('Categorical columns encoding completed')

        preprocessor = ColumnTransformer(
            [
                ('numerical_pipeline', numerical_pipeline, numerical_columns),
                (
                    'categorical_pipeline',
                    categorical_pipeline,
                    categorical_columns,
                ),
            ]
        )

        return preprocessor

    @logger.catch
    def initiate_data_transformation(
        self, train_path: Path, test_path: Path
    ) -> Tuple[Any, Any, Path]:
        """
        This method is responsible for iniate data transformation
        """
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logger.success('Read train and test data completed')

        logger.info('Obtaing pre processing object ...')
        preprocessor = self.get_data_transformer_object()

        target_column = 'math_score'
        numerical_columns = ['writing_score', 'reading_score']

        input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
        target_feature_train_df = train_df[target_column]

        input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
        target_feature_test_df = test_df[target_column]

        logger.info(
            'Applying preporcessing object on training dataframe and testing dataframe ...'
        )

        input_feature_train_arr = preprocessor.fit_transform(
            input_feature_train_df
        )
        input_feature_test_arr = preprocessor.transform(input_feature_test_df)

        train_arr = np.c_[
            input_feature_train_arr, np.array(target_feature_train_df)
        ]
        test_arr = np.c_[
            input_feature_test_arr, np.array(target_feature_test_df)
        ]

        save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessor,
        )

        logger.success('Preprocessor Saved!')

        return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path,
        )
