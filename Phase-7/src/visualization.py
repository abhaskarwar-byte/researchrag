import os
import matplotlib.pyplot as plt


REPORT_DIR = "reports"
FIGURE_DIR = os.path.join(REPORT_DIR, "figures")


# =====================================
# CREATE OUTPUT DIRECTORY
# =====================================

def create_output_directory():

    os.makedirs(
        FIGURE_DIR,
        exist_ok=True
    )


# =====================================
# GRAPH STATISTICS
# =====================================

def plot_graph_statistics(graph_metrics):

    create_output_directory()

    labels = [
        "Papers",
        "Parent Chunks",
        "Child Chunks",
        "Entities",
        "Relationships"
    ]

    values = [
        graph_metrics["papers"],
        graph_metrics["parent_chunks"],
        graph_metrics["child_chunks"],
        graph_metrics["entities"],
        graph_metrics["relationships"]
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(labels, values)

    plt.title("Knowledge Graph Statistics")

    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "graph_statistics.png"
        )
    )

    plt.close()


# =====================================
# RETRIEVAL METRICS
# =====================================

def plot_retrieval_metrics(metrics):

    create_output_directory()

    labels = [
        "Entity Precision",
        "Entity Recall",
        "Entity F1",
        "Relationship Precision",
        "Relationship Recall",
        "Relationship F1"
    ]

    values = [

        metrics["entity_precision"],

        metrics["entity_recall"],

        metrics["entity_f1"],

        metrics["relationship_precision"],

        metrics["relationship_recall"],

        metrics["relationship_f1"]

    ]

    plt.figure(figsize=(9, 5))

    plt.bar(labels, values)

    plt.ylim(0, 1)

    plt.ylabel("Score")

    plt.title("Average Retrieval Metrics")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "retrieval_metrics.png"
        )
    )

    plt.close()


# =====================================
# ANSWER METRICS
# =====================================

def plot_answer_metrics(metrics):

    create_output_directory()

    labels = [

        "Exact Match",

        "Token F1",

        "Entity F1",

        "Relationship F1"

    ]

    values = [

        metrics["exact_match"],

        metrics["token_f1"],

        metrics["entity_f1"],

        metrics["relationship_f1"]

    ]

    plt.figure(figsize=(8, 5))

    plt.bar(labels, values)

    plt.ylim(0, 1)

    plt.ylabel("Score")

    plt.title("Average Answer Metrics")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "answer_metrics.png"
        )
    )

    plt.close()


# =====================================
# QUESTION TYPES
# =====================================

def plot_question_type_distribution(dataset):

    create_output_directory()

    counts = {}

    for question in dataset:

        question_type = question["question_type"]

        counts[question_type] = (

            counts.get(

                question_type,

                0

            ) + 1

        )

    plt.figure(figsize=(6, 6))

    plt.pie(

        counts.values(),

        labels=counts.keys(),

        autopct="%1.1f%%"

    )

    plt.title(

        "Question Type Distribution"

    )

    plt.savefig(

        os.path.join(

            FIGURE_DIR,

            "question_types.png"

        )

    )

    plt.close()


# =====================================
# DIFFICULTY DISTRIBUTION
# =====================================

def plot_difficulty_distribution(dataset):

    create_output_directory()

    counts = {}

    for question in dataset:

        difficulty = question["difficulty"]

        counts[difficulty] = (

            counts.get(

                difficulty,

                0

            ) + 1

        )

    plt.figure(figsize=(6, 5))

    plt.bar(

        counts.keys(),

        counts.values()

    )

    plt.title(

        "Difficulty Distribution"

    )

    plt.ylabel(

        "Questions"

    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_DIR,

            "difficulty_distribution.png"

        )

    )

    plt.close()