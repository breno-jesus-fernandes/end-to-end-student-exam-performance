from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from app.components.data_transformation import DataTransformation


@dataclass
class DataIngestionConfig:
    train_data_path: Path = Path('artifacts/train.csv')
    test_data_path: Path = Path('artifacts/test.csv')
    raw_data_path: Path = Path('artifacts/data.csv')


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    @logger.catch
    def initiate_data_ingestion(self) -> Tuple[Path, Path]:

        logger.info('Entered tha data ingestion method ...')

        logger.info('Reading the dataset  as dataframe using pandas ...')
        df = pd.read_csv('notebook/data/stud.csv')

        logger.info('Creating train data train dir if not exists ...')
        self.ingestion_config.train_data_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        logger.info('Saving the data source in raw destination ...')
        df.to_csv(
            self.ingestion_config.raw_data_path, index=False, header=True
        )

        logger.info('Training test split initiated')
        train_set, test_set = train_test_split(
            df, test_size=0.2, random_state=1982
        )

        logger.info('Saving the train and test')
        train_set.to_csv(
            self.ingestion_config.train_data_path, index=False, header=True
        )
        test_set.to_csv(
            self.ingestion_config.test_data_path, index=False, header=True
        )

        logger.success('Ingestion of data is completed')

        return (
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path,
        )


if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
