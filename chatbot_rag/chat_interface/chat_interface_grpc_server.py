from concurrent import futures

import chatbot_pb2
import chatbot_pb2_grpc
import grpc
from chatbot import ChatBot


class ChatServiceServicer(chatbot_pb2_grpc.ChatServiceServicer):
    def __init__(self):
        self.chatbot = ChatBot()

    def Answer(self, request, context):
        chat_history = []
        input_query = ""
        categories = request.categories
        user_metadata = request.user_metadata
        if request.model_name != self.chatbot.reranker_name:
            self.chatbot.reload_reranker_model(request.model_name)

        last_message = ""
        last_sender = ""
        for i, message in enumerate(request.messages):
            if last_sender != message.sender and last_sender != "":
                chat_history.append((last_sender, last_message))
                last_message = ""

            last_message += " " + message.message
            last_sender = message.sender

        input_query = last_message
        try:
            answer, used_documents = self.chatbot.answer(
                input_query, chat_history, categories, context, **user_metadata
            )
        except TimeoutError as e:
            context.abort(
                grpc.StatusCode.DEADLINE_EXCEEDED,
                f"Timeout occurred in answer method: {str(e)}",
            )
        used_categories = []
        used_document_ids = []
        for doc in used_documents:
            used_document_ids.append(doc.metadata.get("uuid", []))
            used_categories.extend(doc.metadata.get("categories", []))

        return chatbot_pb2.MessageResponse(
            response_message=answer,
            used_categories=list(set(used_categories)),
            used_document_ids=list(set(used_document_ids)),
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chatbot_pb2_grpc.add_ChatServiceServicer_to_server(ChatServiceServicer(), server)
    server.add_insecure_port("0.0.0.0:50055")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
