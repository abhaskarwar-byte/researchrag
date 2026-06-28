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

    print(
        "\nResetting:"
    )

    print(
        os.path.abspath(
            CHUNKS_PATH
        )
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


if __name__ == "__main__":

    reset_database()

    print(
        "\nDatabase reset successfully."
    )