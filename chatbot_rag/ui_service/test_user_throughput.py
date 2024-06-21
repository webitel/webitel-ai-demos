import grpc
import asyncio
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import chatbot_pb2
import chatbot_pb2_grpc

# gRPC channel setup
channel = grpc.insecure_channel('localhost:50055')
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
