from loguru import logger
import os
from concurrent import futures

import grpc
import vector_db_pb2
import vector_db_pb2_grpc
from vector_db_weaviate import VectorDatabase

logger.add(
    "vector_db_interface.log",
    level="DEBUG",
    rotation="50 MB",
    backtrace=True,
    diagnose=True,
)

host = os.getenv("HOST")
http_port = int(os.getenv("HTTP_PORT"))


class VectorDBServiceServicer(vector_db_pb2_grpc.VectorDBServiceServicer):
    def __init__(self):
        self.db = VectorDatabase(host=host, port=http_port)

    def AddArticles(self, request, context):
        logger.debug("Received request: {request}".format(request=request))
        contents = []
        categories = []
        for article in request.articles:
            contents.append(article.content)
            categories.append(list(article.categories))

        message, new_ids = self.db.insert(contents, categories)
        return vector_db_pb2.AddArticlesResponse(id=new_ids, response_message=message)

    def GetArticles(self, request, context):
        logger.debug("Received request: {request}".format(request=request))
        res = self.db.get_articles(request.id, request.categories)
        articles = []
        for article in res:
            articles.append(
                vector_db_pb2.Article(
                    id=article["id"],
                    content=article["content"],
                    categories=article["categories"],
                )
            )
        return vector_db_pb2.GetArticlesResponse(articles=articles)

    def RemoveArticles(self, request, context):
        logger.debug("Received request: {request}".format(request=request))
        message = self.db.remove(request.id)
        return vector_db_pb2.RemoveArticlesResponse(
            id=request.id, response_message=message
        )

    def UpdateArticles(self, request, context):
        logger.debug("Received request: {request}".format(request=request))
        ids = []
        contents = []
        categories = []
        for article in request.articles:
            ids.append(article.id)
            contents.append(article.content)
            categories.append(article.categories)
        message = self.db.upsert(ids, contents, categories)

        return vector_db_pb2.UpdateArticlesResponse(response_message=message)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vector_db_pb2_grpc.add_VectorDBServiceServicer_to_server(
        VectorDBServiceServicer(), server
    )
    server.add_insecure_port("0.0.0.0:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
