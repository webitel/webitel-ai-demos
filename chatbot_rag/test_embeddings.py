import requests
import time

# without threadpool
# python test_embeddings.py
# Average response time: 1.3756176590919496 seconds
# with threadpool
# python test_embeddings.py
# Average response time: 0.4133135895729065 seconds

iterations = 100
total_time = 0
start_time = time.time()

# def make_request():
#     response = requests.post("http://localhost:8000/embeddings", json={"text": 'Як придбати облігації?'})
#     return response
# with ThreadPoolExecutor(max_workers=10) as executor:
#     futures = [executor.submit(make_request) for _ in range(iterations)]

# for future in as_completed(futures):
#     res = future.result()

# end_time = time.time()

# average_time = (end_time-start_time) / iterations
# print(f"Average response time with threads: {average_time} seconds")

total_time = 0
start_time = time.time()

for _ in range(iterations):
    response = requests.post(
        "http://localhost:8000/embeddings", json={"text": "Як придбати облігації?"}
    )
end_time = time.time()

average_time = (end_time - start_time) / iterations
print(f"Average response time without: {average_time} seconds")
