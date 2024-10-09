from rank_bm25 import BM25Okapi
from rapidfuzz import process
from sklearn.metrics.pairwise import cosine_similarity
import requests
from langchain_weaviate import WeaviateVectorStore
from datetime import datetime
import time
import os

embedding_url = os.getenv("EMBEDDING_URL", "http://embedding-service:8000/embeddings")


def get_ukrainian_date(date_str):
    # Ukrainian month names
    months = [
        "січня",
        "лютого",
        "березня",
        "квітня",
        "травня",
        "червня",
        "липня",
        "серпня",
        "вересня",
        "жовтня",
        "листопада",
        "грудня",
    ]

    # Ukrainian hour names
    hour_names = [
        "дванадцятій",
        "першій",
        "другій",
        "третій",
        "четвертій",
        "п'ятій",
        "шостій",
        "сьомій",
        "восьмій",
        "дев'ятій",
        "десятій",
        "одинадцятій",
    ]

    # Parse the input date
    date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M")

    # Extract components
    day = date_obj.day
    month = date_obj.month
    hour = date_obj.hour

    # Convert day and hour to Ukrainian
    day_text = f"{day}-го" if day > 1 else "першого"  # For day "1" should be "першого"
    month_text = months[month - 1]
    hour_text = hour_names[hour % 12]  # Convert 24-hour format to 12-hour format
    if hour >= 12:
        hour_text += " вечора"
    else:
        hour_text += " ранку"

    # Construct the final Ukrainian date string
    ukrainian_date = f"{day_text} {month_text} о {hour_text}"

    return ukrainian_date


def date_selector(delivery_time, all_times):
    print(delivery_time, type(delivery_time))

    delivery_time = datetime.fromisoformat(delivery_time)
    # Convert all strings in all_times to datetime objects
    all_times_dt = [datetime.fromisoformat(time_str) for time_str in all_times]

    # Find the time in all_times that is closest to the delivery_time
    closest_time = min(all_times_dt, key=lambda x: abs(x - delivery_time))

    # Return the closest time formatted back to string (if needed)
    return get_ukrainian_date(closest_time.strftime("%Y-%m-%d %H:%M")), closest_time


def select_bm25(requested, possibilities):
    final_products = []

    tokenized_corpus = [doc.split(" ") for doc in possibilities]
    bm25 = BM25Okapi(tokenized_corpus)
    for request in requested:
        tokenized_query = request.split(" ")
        best_matches = bm25.get_top_n(tokenized_query, possibilities, n=5)
        print(f"best_matches for  {request} : ", best_matches)
        best_match = process.extractOne(request, possibilities)
        final_products.append(best_match[0])
    return final_products


def select_best_matches(
    requested_products: list[str],
    product_names: list[str],
    product_embeddings: list[list[float]],
):
    final_products = []
    # tokenized_corpus = [doc.split(" ") for doc in product_names]
    # bm25 = BM25Okapi(tokenized_corpus)
    for requested_product in requested_products:
        # tokenized_query = requested_product.split(" ")
        # best_matches = bm25.get_top_n(tokenized_query, product_names, n=5)
        # print(f"best_matches for  {requested_product} : ", best_matches)
        # best_match = process.extractOne(requested_product, product_names)

        # final_products.append(best_match[0])
        target_embedding = get_embedding(requested_product)
        best_similarity = -1
        best_match = None
        for product_name, product_embedding in zip(product_names, product_embeddings):
            # Calculate cosine similarity between target_embedding and product_embedding
            similarity = cosine_similarity([target_embedding], [product_embedding])[0][
                0
            ]
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = product_name
        print(f"Embedding search {requested_product}")
        print(best_match)
        print(best_similarity)
        final_products.append(best_match)

    return final_products


def get_embedding(text: str):
    response = requests.post(
        embedding_url, json={"text": text, "task": "text-matching"}
    )
    response_body = response.json()
    return response_body["embedding"]


def select_best_match_in_db(client, requested_products, index_name="ProductList"):
    connection_start = time.time()
    retriever = WeaviateVectorStore(
        client,
        attributes=["categories"],
        text_key="content",
        index_name=index_name,
        embedding=get_embedding,
    )
    connection_end = time.time()
    print(f"Connection time: {connection_end - connection_start}")
    best_matches = []
    alpha = 0.0
    if index_name != "ProductList":
        alpha = 0.0
    total_time_embedding = 0
    total_time_search = 0
    for product in requested_products:
        embedding_start = time.time()
        vector = get_embedding(product)
        embedding_end = time.time()
        total_time_embedding += embedding_end - embedding_start
        kwargs = {
            "return_uuids": True,
            "vector": vector,
            "alpha": alpha,  # 1 - pure vector search, 0 - pure keyword search,
        }
        search_start = time.time()
        res = retriever.similarity_search(query=product, k=1, **kwargs)
        search_end = time.time()
        best_matches.append(res[0].page_content)
        total_time_search += search_end - search_start
    print(f"Total time embedding: {total_time_embedding}")
    print(f"Total time search: {total_time_search}")
    return best_matches


def select_best_match_in_db2(client, requested_products, index_name="ProductList"):
    connection_start = time.time()
    retriever = WeaviateVectorStore(
        client,
        attributes=["categories"],
        text_key="content",
        index_name=index_name,
        embedding=get_embedding,
    )
    connection_end = time.time()
    print(f"Connection time: {connection_end - connection_start}")
    alpha = 1
    if index_name != "ProductList":
        alpha = 1
    total_time_embedding = 0
    total_time_search = 0
    best_matches = {}
    for product in requested_products:
        embedding_start = time.time()
        vector = get_embedding(product)
        embedding_end = time.time()
        total_time_embedding += embedding_end - embedding_start
        kwargs = {
            "return_uuids": True,
            "vector": vector,
            "alpha": alpha,  # 1 - pure vector search, 0 - pure keyword search,
        }
        search_start = time.time()
        res = retriever.similarity_search(query=product, k=5, **kwargs)
        search_end = time.time()
        print(res)
        best_matches[product] = [
            (res[0].page_content, res[0].metadata["price"]),
            (res[1].page_content, res[1].metadata["price"]),
            (res[2].page_content, res[2].metadata["price"]),
            (res[3].page_content, res[3].metadata["price"]),
            (res[4].page_content, res[4].metadata["price"]),
        ]

        total_time_search += search_end - search_start
    print(f"Total time embedding: {total_time_embedding}")
    print(f"Total time search: {total_time_search}")
    print(best_matches)
    return best_matches
