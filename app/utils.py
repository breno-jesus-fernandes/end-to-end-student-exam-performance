from pathlib import Path
from typing import Any

import dill
from loguru import logger


@logger.catch
def save_object(file_path: Path, obj: Any):
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'wb') as file:
        dill.dump(obj, file)
