from tavily import TavilyClient

from src.config import (
    TAVILY_API_KEY
)


client = TavilyClient(
    api_key=TAVILY_API_KEY
)


def web_search(
    question,
    max_results=5
):

    response = client.search(

        query=question,

        search_depth="advanced",

        max_results=max_results,

        include_answer=False,

        include_raw_content=False
    )

    web_chunks = []

    for i, result in enumerate(
        response["results"],
        start=1
    ):

        web_chunks.append(

            {
                "source": result["url"],

                "page": 0,

                "chunk_number": i,

                "parent_id": -1,

                "chunk_type": "web",

                "text": result["content"]
            }

        )

    return web_chunks


if __name__ == "__main__":

    results = web_search(
        "Why is the Transformer paper called Attention Is All You Need?"
    )

    print(
        "\nWEB RESULTS\n"
    )

    for chunk in results:

        print(
            chunk["source"]
        )

        print(
            chunk["text"][:300]
        )

        print("-" * 60)