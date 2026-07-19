import os
import json

from src.graph_evaluator import evaluate_dataset
from src.retrieval_evaluator import evaluate_retrieval
from src.answer_generator import generate_answer_from_retrieval

from src.answer_evaluator import (
    evaluate_answer,
    evaluate_answers
)

from src.graph_metrics import graph_statistics

from src.visualization import (
    plot_graph_statistics,
    plot_retrieval_metrics,
    plot_answer_metrics,
    plot_question_type_distribution,
    plot_difficulty_distribution
)

from src.report_generator import generate_report


# =====================================
# CHECKPOINT DIRECTORY
# =====================================

REPORT_DIR = "reports"

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# =====================================
# JSON HELPERS
# =====================================

def save_json(filename, data):

    path = os.path.join(
        REPORT_DIR,
        filename
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


def load_json(filename):

    path = os.path.join(
        REPORT_DIR,
        filename
    )

    if not os.path.exists(path):
        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# =====================================
# PIPELINE
# =====================================

def run_pipeline():

    print("\n======================================")
    print("KNOWLEDGE GRAPH RAG EVALUATION")
    print("======================================")

    # ---------------------------------
    # GRAPH EVALUATION
    # ---------------------------------

    graph_results = load_json(
        "graph_results.json"
    )

    graph_metrics = load_json(
        "graph_metrics.json"
    )

    if graph_results is None or graph_metrics is None:

        print("\nRunning graph evaluation...")

        graph_metrics = graph_statistics()

        graph_results = evaluate_dataset()

        save_json(
            "graph_metrics.json",
            graph_metrics
        )

        save_json(
            "graph_results.json",
            graph_results
        )

    else:

        print("\nLoaded graph checkpoint.")

    # ---------------------------------
    # RETRIEVAL EVALUATION
    # ---------------------------------

    retrieval_results = load_json(
        "retrieval_results.json"
    )

    retrieval_summary = load_json(
        "retrieval_summary.json"
    )

    if retrieval_results is None or retrieval_summary is None:

        print("\nRunning retrieval evaluation...")

        retrieval_results, retrieval_summary = evaluate_retrieval(
            graph_results
        )

        save_json(
            "retrieval_results.json",
            retrieval_results
        )

        save_json(
            "retrieval_summary.json",
            retrieval_summary
        )

    else:

        print("\nLoaded retrieval checkpoint.")

    # ---------------------------------
    # ANSWER CHECKPOINT
    # ---------------------------------

    answer_results = load_json(
        "answer_checkpoint.json"
    )

    if answer_results is None:

        answer_results = []

    completed_questions = {

        result.get("question")

        for result in answer_results

        if "question" in result

    }

    print(
        f"\nLoaded {len(answer_results)} completed answers."
    )

    # ---------------------------------
    # ANSWER GENERATION
    # ---------------------------------

    try:

        for benchmark in retrieval_results:

            question = benchmark["question"]

            if question in completed_questions:

                print(
                    f"Skipping: {question[:60]}..."
                )

                continue

            print(
                f"\nGenerating Answer ({len(answer_results)+1}/{len(retrieval_results)})"
            )

            response = generate_answer_from_retrieval(

                question=question,

                vector_context=benchmark["retrieved_vector"],

                graph_context=benchmark["retrieved_graph"]

            )

            metrics = evaluate_answer(

                question=question,

                reference_answer=benchmark["reference_answer"],

                generated_answer=response["answer"],

                expected_entities=benchmark["expected_entities"],

                retrieved_entities=[

                    {
                        "name": entity["entity"]
                    }

                    for entity in benchmark["retrieved_graph"]

                ],

                expected_relationships=benchmark["expected_relationships"],

                retrieved_relationships=benchmark["retrieved_relationships"]

            )

            answer_results.append(metrics)

            save_json(
                "answer_checkpoint.json",
                answer_results
            )

            print("Checkpoint saved.")

    except Exception as e:

        print("\n======================================")
        print("ANSWER GENERATION INTERRUPTED")
        print("======================================")
        print(e)

        if answer_results:

            answer_summary = evaluate_answers(
                answer_results
            )

            save_json(
                "answer_summary.json",
                answer_summary
            )

        else:

            answer_summary = {}

        print("\nGenerating partial visualizations...")

        plot_graph_statistics(
            graph_metrics
        )

        plot_retrieval_metrics(
            retrieval_summary
        )

        if answer_summary:

            plot_answer_metrics(
                answer_summary
            )

        plot_question_type_distribution(
            retrieval_results
        )

        plot_difficulty_distribution(
            retrieval_results
        )

        print("\nGenerating partial report...")

        generate_report(

            graph_metrics,

            retrieval_summary,

            answer_summary

        )

        print("\n======================================")
        print("PARTIAL EVALUATION COMPLETE")
        print("======================================")
        print(
            f"Checkpoint saved with {len(answer_results)} completed answers."
        )

        return

    # ---------------------------------
    # FINAL ANSWER SUMMARY
    # ---------------------------------

    answer_summary = evaluate_answers(
        answer_results
    )

    save_json(
        "answer_summary.json",
        answer_summary
    )

    # ---------------------------------
    # VISUALIZATIONS
    # ---------------------------------

    print("\nGenerating visualizations...")

    plot_graph_statistics(
        graph_metrics
    )

    plot_retrieval_metrics(
        retrieval_summary
    )

    plot_answer_metrics(
        answer_summary
    )

    plot_question_type_distribution(
        retrieval_results
    )

    plot_difficulty_distribution(
        retrieval_results
    )

    # ---------------------------------
    # REPORT
    # ---------------------------------

    print("\nGenerating report...")

    generate_report(

        graph_metrics,

        retrieval_summary,

        answer_summary

    )

    print("\n======================================")
    print("EVALUATION COMPLETE")
    print("======================================")

    print("\nSaved files:")

    print("reports/graph_results.json")
    print("reports/graph_metrics.json")
    print("reports/retrieval_results.json")
    print("reports/retrieval_summary.json")
    print("reports/answer_checkpoint.json")
    print("reports/answer_summary.json")
    print("reports/evaluation_report.txt")


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    run_pipeline()