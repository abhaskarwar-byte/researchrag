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

    # ==========================
    # Retrieval Metrics
    # ==========================

    entity_precision = average(
        [r["entity_precision"] for r in retrieval_results]
    )

    entity_recall = average(
        [r["entity_recall"] for r in retrieval_results]
    )

    entity_f1 = average(
        [r["entity_f1"] for r in retrieval_results]
    )

    relationship_precision = average(
        [r["relationship_precision"] for r in retrieval_results]
    )

    relationship_recall = average(
        [r["relationship_recall"] for r in retrieval_results]
    )

    relationship_f1 = average(
        [r["relationship_f1"] for r in retrieval_results]
    )

    # ==========================
    # Answer Metrics
    # ==========================

    exact_match = average(
        [r["exact_match"] for r in answer_results]
    )

    token_f1 = average(
        [r["token_f1"] for r in answer_results]
    )

    answer_entity_f1 = average(
        [r["entity_f1"] for r in answer_results]
    )

    answer_relationship_f1 = average(
        [r["relationship_f1"] for r in answer_results]
    )

    # ==========================
    # Generate Report
    # ==========================

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

        report.write("## Retrieval Evaluation\n\n")

        report.write("### Entity Retrieval\n\n")
        report.write(f"- Precision : {entity_precision:.4f}\n")
        report.write(f"- Recall : {entity_recall:.4f}\n")
        report.write(f"- F1 Score : {entity_f1:.4f}\n\n")

        report.write("### Relationship Retrieval\n\n")
        report.write(f"- Precision : {relationship_precision:.4f}\n")
        report.write(f"- Recall : {relationship_recall:.4f}\n")
        report.write(f"- F1 Score : {relationship_f1:.4f}\n")

        report.write("\n---\n\n")

        report.write("## Answer Evaluation\n\n")

        report.write(f"- Exact Match : {exact_match:.4f}\n")
        report.write(f"- Token F1 : {token_f1:.4f}\n")
        report.write(f"- Entity F1 : {answer_entity_f1:.4f}\n")
        report.write(f"- Relationship F1 : {answer_relationship_f1:.4f}\n")

        report.write("\n---\n\n")

        report.write("## Generated Visualizations\n\n")

        report.write("- reports/figures/graph_statistics.png\n")
        report.write("- reports/figures/retrieval_metrics.png\n")
        report.write("- reports/figures/answer_metrics.png\n")
        report.write("- reports/figures/question_types.png\n")
        report.write("- reports/figures/difficulty_distribution.png\n")

        report.write("\n---\n\n")

        report.write("## Summary\n\n")

        report.write(
            "The Knowledge Graph RAG evaluation was performed using a "
            "benchmark dataset generated from the ingested research papers. "
            "The system was evaluated in three stages: knowledge graph quality, "
            "retrieval performance, and answer quality. The generated charts "
            "provide a visual overview of graph statistics, retrieval metrics, "
            "answer metrics, benchmark question types, and difficulty distribution."
        )

    print(f"\nReport saved to: {REPORT_FILE}")