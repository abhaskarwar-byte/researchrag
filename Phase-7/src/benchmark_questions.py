import json
import os


# =====================================================
# CONFIGURATION
# =====================================================

BENCHMARK_FILE = os.path.join(
    "benchmarks",
    "benchmark_dataset.json"
)


# =====================================================
# LOAD DATASET
# =====================================================

def load_benchmark_dataset():

    if not os.path.exists(BENCHMARK_FILE):

        raise FileNotFoundError(

            f"Benchmark dataset not found:\n{BENCHMARK_FILE}"

        )

    with open(

        BENCHMARK_FILE,

        "r",

        encoding="utf-8"

    ) as file:

        dataset = json.load(file)

    return dataset


# =====================================================
# GET ALL QUESTIONS
# =====================================================

def get_all_questions():

    return load_benchmark_dataset()


# =====================================================
# GET QUESTION BY ID
# =====================================================

def get_question_by_id(question_id):

    dataset = load_benchmark_dataset()

    for sample in dataset:

        if sample["id"] == question_id:

            return sample

    return None


# =====================================================
# GET QUESTIONS BY PAPER
# =====================================================

def get_questions_by_paper(paper_name):

    dataset = load_benchmark_dataset()

    return [

        sample

        for sample in dataset

        if sample["paper"] == paper_name

    ]


# =====================================================
# GET QUESTIONS BY DIFFICULTY
# =====================================================

def get_questions_by_difficulty(difficulty):

    dataset = load_benchmark_dataset()

    return [

        sample

        for sample in dataset

        if sample["difficulty"].lower()

        == difficulty.lower()

    ]


# =====================================================
# GET QUESTIONS BY TYPE
# =====================================================

def get_questions_by_type(question_type):

    dataset = load_benchmark_dataset()

    return [

        sample

        for sample in dataset

        if sample["question_type"].lower()

        == question_type.lower()

    ]


# =====================================================
# GET QUESTIONS BY PAGE
# =====================================================

def get_questions_by_page(

        paper_name,

        page

):

    dataset = load_benchmark_dataset()

    return [

        sample

        for sample in dataset

        if (

            sample["paper"] == paper_name

            and

            sample["page"] == page

        )

    ]


# =====================================================
# DATASET STATISTICS
# =====================================================

def dataset_statistics():

    dataset = load_benchmark_dataset()

    papers = {

        sample["paper"]

        for sample in dataset

    }

    pages = {

        (

            sample["paper"],

            sample["page"]

        )

        for sample in dataset

    }

    print("\n================================")

    print("BENCHMARK DATASET")

    print("================================")

    print(

        f"Total Questions : {len(dataset)}"

    )

    print(

        f"Research Papers : {len(papers)}"

    )

    print(

        f"Pages Covered   : {len(pages)}"

    )

    print(

        f"Papers          :"

    )

    for paper in sorted(papers):

        count = len(

            get_questions_by_paper(

                paper

            )

        )

        print(

            f"  {paper} : {count}"

        )


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    dataset_statistics()