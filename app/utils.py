from pathlib import Path
from typing import Any, Dict

import dill
from loguru import logger
from sklearn.metrics import r2_score


@logger.catch
def save_object(file_path: Path, obj: Any):
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'wb') as file:
        dill.dump(obj, file)


@logger.catch
def evaluate_models(
    X_train, y_train, X_test, y_test, models: Dict[str, Any]
) -> Dict[str, float]:
    """
    This function is responsible for evaluate de r2 score from different ML models
    """

    report: Dict[str, float] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_model_score = r2_score(y_train, y_train_pred)
        test_model_score = r2_score(y_test, y_test_pred)

        logger.info(f'Model: {name}, R2 SCORE: {test_model_score:.2f}')
        report[name] = float(test_model_score)

    return report
