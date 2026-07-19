import os


# =====================================
# REPORT GENERATOR
# =====================================

def generate_report(
    graph_metrics,
    retrieval_results,
    answer_results,
    output_file="reports/evaluation_report.txt"
):
    """
    Generates the final evaluation report.

    Parameters
    ----------
    graph_metrics : dict
        Graph statistics.

    retrieval_results : dict
        Average retrieval metrics.

    answer_results : dict
        Average answer metrics.
        May be empty if answer generation failed.

    output_file : str
        Output report path.
    """

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as report:

        report.write("=" * 60 + "\n")
        report.write("KNOWLEDGE GRAPH RAG EVALUATION REPORT\n")
        report.write("=" * 60 + "\n\n")

        # ==================================================
        # GRAPH STATISTICS
        # ==================================================

        report.write("GRAPH STATISTICS\n")
        report.write("-" * 40 + "\n")

        for key, value in graph_metrics.items():
            report.write(f"{key:<30}: {value}\n")

        report.write("\n")

        # ==================================================
        # RETRIEVAL METRICS
        # ==================================================

        report.write("RETRIEVAL METRICS\n")
        report.write("-" * 40 + "\n")

        report.write(
            f"Entity Precision       : {retrieval_results['entity_precision']:.3f}\n"
        )
        report.write(
            f"Entity Recall          : {retrieval_results['entity_recall']:.3f}\n"
        )
        report.write(
            f"Entity F1              : {retrieval_results['entity_f1']:.3f}\n\n"
        )

        report.write(
            f"Relationship Precision : {retrieval_results['relationship_precision']:.3f}\n"
        )
        report.write(
            f"Relationship Recall    : {retrieval_results['relationship_recall']:.3f}\n"
        )
        report.write(
            f"Relationship F1        : {retrieval_results['relationship_f1']:.3f}\n"
        )

        report.write("\n")

        # ==================================================
        # ANSWER QUALITY
        # ==================================================

        report.write("ANSWER QUALITY\n")
        report.write("-" * 40 + "\n")

        if answer_results:

            report.write(
                f"Exact Match            : {answer_results['exact_match']:.3f}\n"
            )

            report.write(
                f"Token Precision        : {answer_results['token_precision']:.3f}\n"
            )

            report.write(
                f"Token Recall           : {answer_results['token_recall']:.3f}\n"
            )

            report.write(
                f"Token F1               : {answer_results['token_f1']:.3f}\n\n"
            )

            report.write(
                f"Entity Precision       : {answer_results['entity_precision']:.3f}\n"
            )

            report.write(
                f"Entity Recall          : {answer_results['entity_recall']:.3f}\n"
            )

            report.write(
                f"Entity F1              : {answer_results['entity_f1']:.3f}\n\n"
            )

            report.write(
                f"Relationship Precision : {answer_results['relationship_precision']:.3f}\n"
            )

            report.write(
                f"Relationship Recall    : {answer_results['relationship_recall']:.3f}\n"
            )

            report.write(
                f"Relationship F1        : {answer_results['relationship_f1']:.3f}\n"
            )

        else:

            report.write(
                "Answer evaluation could not be completed because the LLM API quota was exhausted.\n"
            )

        report.write("\n")

        # ==================================================
        # GENERATED VISUALIZATIONS
        # ==================================================

        report.write("GENERATED VISUALIZATIONS\n")
        report.write("-" * 40 + "\n")

        figures = [
            "graph_statistics.png",
            "retrieval_metrics.png",
            "answer_metrics.png",
            "question_types.png",
            "difficulty_distribution.png",
        ]

        for figure in figures:
            report.write(f"- {figure}\n")

        report.write("\n")
        report.write("=" * 60 + "\n")
        report.write("END OF REPORT\n")
        report.write("=" * 60 + "\n")

    print("\n======================================")
    print("REPORT GENERATED")
    print("======================================")
    print(f"Saved to: {output_file}")