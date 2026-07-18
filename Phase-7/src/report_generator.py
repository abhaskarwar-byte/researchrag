import os


REPORT_DIR = "reports"
REPORT_FILE = os.path.join(REPORT_DIR, "evaluation_report.md")


def create_report_directory():
    """
    Creates the reports directory if it doesn't exist.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)


def average(values):
    """
    Returns the average of a list.
    """
    if not values:
        return 0.0

    return round(sum(values) / len(values), 4)


def generate_report(
    graph_metrics,
    retrieval_results,
    answer_results
):
    """
    Generates a Markdown evaluation report.

    Parameters
    ----------
    graph_metrics : dict

    retrieval_results : list

    answer_results : list
    """

    create_report_directory()

    retrieval_precision = average(
        [r["precision"] for r in retrieval_results]
    )

    retrieval_recall = average(
        [r["recall"] for r in retrieval_results]
    )

    retrieval_f1 = average(
        [r["f1"] for r in retrieval_results]
    )

    exact_match = average(
        [r["exact_match"] for r in answer_results]
    )

    token_f1 = average(
        [r["token_f1"] for r in answer_results]
    )

    entity_f1 = average(
        [r["entity_f1"] for r in answer_results]
    )

    relationship_f1 = average(
        [r["relationship_f1"] for r in answer_results]
    )

    with open(REPORT_FILE, "w", encoding="utf-8") as report:

        report.write("# Research RAG Evaluation Report\n\n")

        report.write("---\n\n")

        report.write("## Knowledge Graph Statistics\n\n")

        report.write(f"- Papers : {graph_metrics['papers']}\n")
        report.write(f"- Parent Chunks : {graph_metrics['parent_chunks']}\n")
        report.write(f"- Child Chunks : {graph_metrics['child_chunks']}\n")
        report.write(f"- Entities : {graph_metrics['entities']}\n")
        report.write(f"- Relationships : {graph_metrics['relationships']}\n")

        report.write("\n---\n\n")

        report.write("## Retrieval Performance\n\n")

        report.write(f"- Precision : {retrieval_precision:.4f}\n")
        report.write(f"- Recall : {retrieval_recall:.4f}\n")
        report.write(f"- F1 Score : {retrieval_f1:.4f}\n")

        report.write("\n---\n\n")

        report.write("## Answer Evaluation\n\n")

        report.write(f"- Exact Match : {exact_match:.4f}\n")
        report.write(f"- Token F1 : {token_f1:.4f}\n")
        report.write(f"- Entity F1 : {entity_f1:.4f}\n")
        report.write(f"- Relationship F1 : {relationship_f1:.4f}\n")

        report.write("\n---\n\n")

        report.write("## Generated Visualizations\n\n")

        report.write("- graph_statistics.png\n")
        report.write("- retrieval_metrics.png\n")
        report.write("- answer_metrics.png\n")
        report.write("- question_types.png\n")
        report.write("- difficulty_distribution.png\n")

        report.write("\n---\n\n")

        report.write("## Summary\n\n")

        report.write(
            "This report summarizes the performance of the Knowledge Graph "
            "based Retrieval-Augmented Generation (KG-RAG) system. "
            "The evaluation includes graph statistics, retrieval quality, "
            "and answer quality using benchmark questions.\n"
        )

    print(f"\nReport saved to: {REPORT_FILE}")