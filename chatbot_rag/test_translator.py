import requests
import time
#  python test_translator.py 
# Average response time: 0.811565306186676 seconds
iterations = 100
total_time = 0

for _ in range(iterations):
    start_time = time.time()
    
    response = requests.post("http://localhost:8251/translate", json={"text": 'Як придбати облігації?'})
    print(response.json())
    end_time = time.time()
    execution_time = end_time - start_time
    total_time += execution_time

average_time = total_time / iterations
print(f"Average response time: {average_time} seconds")
