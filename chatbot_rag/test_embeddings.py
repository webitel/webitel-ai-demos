import requests
import time
# python test_embeddings.py 
# Average response time: 1.3756176590919496 seconds
iterations = 100
total_time = 0

for _ in range(iterations):
    start_time = time.time()
    
    response = requests.post("http://localhost:8000/embeddings", json={"text": 'Як придбати облігації?'})

    end_time = time.time()
    execution_time = end_time - start_time
    total_time += execution_time

average_time = total_time / iterations
print(f"Average response time: {average_time} seconds")
