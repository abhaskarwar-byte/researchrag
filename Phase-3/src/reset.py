import json
import os


DATA_DIR = "data"
CHUNKS_PATH = os.path.join(
    DATA_DIR,
    "chunks.json"
)


def reset_database():

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    with open(
        CHUNKS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            [],
            f,
            indent=4
        )

    return True
