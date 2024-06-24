import grpc
import asyncio
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import chatbot_pb2
import chatbot_pb2_grpc

# gRPC channel setup
channel = grpc.insecure_channel('localhost:8081')
stub = chatbot_pb2_grpc.ChatServiceStub(channel)

# Test parameters
user_metadata = {}
categories = ['category']
message_text = 'Як мені купити воду?'

# Function to send gRPC request
def send_message():
    try:
        start = time.time()
        response = stub.Answer(chatbot_pb2.MessageRequest(
            user_metadata=user_metadata, categories=categories, messages=[chatbot_pb2.Message(message=message_text, sender='human')]))
        end = time.time()
        return end - start, response
    except grpc.RpcError as e:
        return None, f"RPC Error: {e}"
    except Exception as e:
        return None, str(e)

async def main():
    num_messages = 22  # Number of messages to send concurrently

    # Use ThreadPoolExecutor to manage concurrent tasks
    with ThreadPoolExecutor(max_workers=num_messages) as executor:
        # Submit tasks to the executor asynchronously
        tasks = [loop.run_in_executor(executor, send_message) for _ in range(num_messages)]

        # Await all tasks to complete concurrently
        results = await asyncio.gather(*tasks)

    all_times = []
    for time_taken, response in results:
        if time_taken is not None:
            print(f"Response received: {response}")
            all_times.append(time_taken)
        else:
            print(f"Error occurred: {response}")

    # Calculate and print statistics
    if all_times:
        print(f"Average time: {np.mean(all_times)} seconds")
        print(f"Max time: {np.max(all_times)} seconds")
        print(f"Min time: {np.min(all_times)} seconds")
        print(f"Std time: {np.std(all_times)} seconds")

if __name__ == '__main__':
    import time
    start_general = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    end_time = time.time()
    
    print(f"Total time taken: {end_time - start_general} seconds")


# with 1 thread and 22 concurrent messages 
# Average time: 43.925138516859576 seconds
# Max time: 78.52218389511108 seconds
# Min time: 4.152154207229614 seconds
# Std time: 21.07320392091566 seconds
# Total time taken: 78.54210901260376 seconds


# with 10 threads and 22 concurrent messages 
# Average time: 16.30094156482003 seconds
# Max time: 22.422840118408203 seconds
# Min time: 4.616497278213501 seconds
# Std time: 4.161973605894377 seconds
# Total time taken: 22.430484294891357 seconds

# with 10 threads and 22 concurrent messages; second run
# Average time: 12.674482800743796 seconds
# Max time: 17.94372844696045 seconds
# Min time: 3.954347848892212 seconds
# Std time: 3.717238810535901 seconds
# Total time taken: 17.96164035797119 seconds

# with 22 threads and 22 concurrent messages 
# Average time: 26.19019732692025 seconds
# Max time: 31.661612033843994 seconds
# Min time: 6.172985553741455 seconds
# Std time: 8.134393090002066 seconds
# Total time taken: 31.690311193466187 seconds

# with 22 threads and 22 concurrent messages; second run
# Average time: 12.871224240823226 seconds
# Max time: 17.083094358444214 seconds
# Min time: 10.55447006225586 seconds
# Std time: 1.1635044961861212 seconds
# Total time taken: 17.11190891265869 seconds

# with 10 threads and 22 concurrent messages; two replicas
# Average time: 11.075560136274857 seconds
# Max time: 18.76432704925537 seconds
# Min time: 4.980943441390991 seconds
# Std time: 4.538760932072202 seconds
# Total time taken: 18.791161060333252 seconds