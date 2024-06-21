import grpc
import vector_db_pb2_grpc
import vector_db_pb2
import chatbot_pb2
import chatbot_pb2_grpc
import time
import numpy as np
from tqdm import tqdm
channel = grpc.insecure_channel('localhost:50055')
stub = chatbot_pb2_grpc.ChatServiceStub(channel)    


user_metadata = {}
categories = ['category']#['']
messages = [chatbot_pb2.Message(message='Як мені купити воду?',sender='human')]

# times = []
# for i in tqdm(range(5)):
#     start = time.time()
#     MessageResponse = stub.Answer(chatbot_pb2.MessageRequest(user_metadata=user_metadata, categories=categories, messages=messages))
#     print(MessageResponse)
#     end = time.time()
#     times.append(end - start)
    
# print("Without timeout")
# print(f"Average time: {sum(times) / len(times)}")
# print(f"Max time: {max(times)}")
# print(f"Std time: {np.std(times)}")


timeout = 1
times = []
messages = [chatbot_pb2.Message(message='Як мені купити воду? timeout',sender='human')]
for i in tqdm(range(5)):
    start = time.time()
    try:
        MessageResponse = stub.Answer(chatbot_pb2.MessageRequest(user_metadata=user_metadata, categories=categories, messages=messages),timeout=1)
        print(MessageResponse)
    except Exception as e:
        print(e)
    end = time.time()
    times.append(end - start)
    
print("With timeout")
print(f"Average time: {sum(times) / len(times)}")
print(f"Max time: {max(times)}")
print(f"Std time: {np.std(times)}")