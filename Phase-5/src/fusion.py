from collections import defaultdict


def rrf_fusion(
    ranked_lists,
    k=60
):

    scores = defaultdict(
        float
    )

    chunk_lookup = {}

    for ranked_list in ranked_lists:

        for rank, (
            score,
            chunk
        ) in enumerate(
            ranked_list,
            start=1
        ):

            parent_id = (
                chunk["parent_id"]
            )

            scores[
                parent_id
            ] += (
                1.0
                /
                (
                    k + rank
                )
            )

            chunk_lookup[
                parent_id
            ] = chunk

    fused_results = []

    for parent_id, rrf_score in (
        scores.items()
    ):

        fused_results.append(
            (
                rrf_score,
                chunk_lookup[
                    parent_id
                ]
            )
        )

    fused_results.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    return fused_results


if __name__ == "__main__":

    query1 = [
        (
            1.5,
            {
                "parent_id": 14
            }
        ),
        (
            1.2,
            {
                "parent_id": 7
            }
        )
    ]

    query2 = [
        (
            1.6,
            {
                "parent_id": 7
            }
        ),
        (
            1.3,
            {
                "parent_id": 14
            }
        )
    ]

    results = rrf_fusion(
        [
            query1,
            query2
        ]
    )

    print(
        "\nFUSED RESULTS:\n"
    )

    for score, chunk in results:

        print(
            chunk["parent_id"],
            round(score, 5)
        )