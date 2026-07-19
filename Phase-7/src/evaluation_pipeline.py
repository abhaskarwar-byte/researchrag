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
# EVALUATION PIPELINE
# =====================================

def run_pipeline():

    print("\n======================================")
    print("KNOWLEDGE GRAPH RAG EVALUATION")
    print("======================================")

    # -----------------------------------
    # Graph Statistics
    # -----------------------------------

    print("\nComputing graph statistics...")

    graph_metrics = graph_statistics()

    # -----------------------------------
    # Graph Retrieval Evaluation
    # -----------------------------------

    print("\nRunning graph retrieval...")

    graph_results = evaluate_dataset()

    # -----------------------------------
    # Retrieval Metrics
    # -----------------------------------

    print("\nComputing retrieval metrics...")

    retrieval_results, retrieval_summary = evaluate_retrieval(graph_results)

    # -----------------------------------
    # Answer Generation
    # -----------------------------------

    print("\nGenerating answers...")

    answer_results = []

    try:

        for benchmark in retrieval_results:

            response = generate_answer_from_retrieval(

                question=benchmark["question"],

                vector_context=benchmark["retrieved_vector"],

                graph_context=benchmark["retrieved_graph"]

            )

            metrics = evaluate_answer(

                question=benchmark["question"],

                reference_answer=benchmark["reference_answer"],

                generated_answer=response["answer"],

                expected_entities=benchmark["expected_entities"],

                retrieved_entities=[
                    {"name": entity["entity"]}
                    for entity in benchmark["retrieved_graph"]
                ],

                expected_relationships=benchmark["expected_relationships"],

                retrieved_relationships=benchmark["retrieved_relationships"]

            )

            answer_results.append(metrics)

    except Exception as e:

        print("\n======================================")
        print("ANSWER GENERATION INTERRUPTED")
        print("======================================")
        print(e)

        if answer_results:
            answer_summary = evaluate_answers(answer_results)
        else:
            answer_summary = {}

        print("\nGenerating partial visualizations...")

        plot_graph_statistics(graph_metrics)

        plot_retrieval_metrics(retrieval_summary)

        if answer_summary:
            plot_answer_metrics(answer_summary)

        plot_question_type_distribution(retrieval_results)

        plot_difficulty_distribution(retrieval_results)

        print("\nGenerating partial report...")

        generate_report(

            graph_metrics,

            retrieval_summary,

            answer_summary

        )

        print("\n======================================")
        print("PARTIAL EVALUATION COMPLETE")
        print("======================================")

        return

    # -----------------------------------
    # Aggregate Answer Metrics
    # -----------------------------------

    answer_summary = evaluate_answers(answer_results)

    # -----------------------------------
    # Visualizations
    # -----------------------------------

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

    # -----------------------------------
    # Report
    # -----------------------------------

    print("\nGenerating report...")

    generate_report(

        graph_metrics,

        retrieval_summary,

        answer_summary

    )

    print("\n======================================")
    print("EVALUATION COMPLETE")
    print("======================================")


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    run_pipeline()