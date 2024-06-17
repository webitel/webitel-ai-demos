import chatbot_pb2
import chatbot_pb2_grpc
import grpc
from concurrent import futures
from chatbot import ChatBot


class ChatServiceServicer(chatbot_pb2_grpc.ChatServiceServicer):
    def __init__(self):
        self.chatbot  = ChatBot()

    def Answer(self, request, context):
        chat_history = []
        input_query = ""
        timeout = request.timeout
        user_metadata = request.user_metadata
        
        last_message = ""
        last_sender = ""
        for i,message in enumerate(request.messages):
            
            if last_sender != message.sender and last_sender != "":
                chat_history.append((last_sender,last_message))
                last_message = ""

            last_message += ' '+ message.message
            last_sender = message.sender

        input_query = last_message
        def execute_chatbot():
            answer, used_docs = self.chatbot.answer(input_query, chat_history, **user_metadata)
            return answer, used_docs
        

        answer, used_documents = execute_chatbot()

        used_categories = []
        used_document_ids = []
        for doc in used_documents:
            used_document_ids.append(doc.metadata.get("uuid",[]))
            used_categories.extend(doc.metadata.get("categories",[]))

        return chatbot_pb2.MessageResponse(response_message=answer,used_categories=list(set(used_categories)),
                                        used_document_ids = list(set(used_document_ids)))


    
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chatbot_pb2_grpc.add_ChatServiceServicer_to_server(ChatServiceServicer(), server)
    server.add_insecure_port('0.0.0.0:50055')
    server.start()
    server.wait_for_termination()
    
if __name__ == '__main__':
    serve()