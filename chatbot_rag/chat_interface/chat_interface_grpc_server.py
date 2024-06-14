import vector_db_pb2
import vector_db_pb2_grpc
import grpc
from concurrent import futures
from vector_db import VectorDatabase


class VectorDBServiceServicer(vector_db_pb2_grpc.VectorDBServiceServicer):
    def __init__(self):
        self.db = VectorDatabase(connection_uri="http://standalone:19530")

    def AddArticles(self, request, context):
        contents = []
        categories = []
        for article in request.articles:
            contents.append(article.content)
            categories.append(article.categories)

        res_insert, new_ids = self.db.insert(contents, categories)

        if res_insert.succ_count == len(contents):
            message = "All articles inserted successfully"
        else:
            message = "Some articles failed to insert"

        return vector_db_pb2.AddArticlesResponse(id=new_ids,response_message=message)

    def GetArticles(self, request, context):
        res = self.db.get_articles(request.id, request.categories)
        articles = []
        for article in res :
            articles.append(vector_db_pb2.Article(id=article['doc_id'], content=article['text'], categories=article['category']))
        return vector_db_pb2.GetArticlesResponse(articles=articles)
    
    def RemoveArticles(self, request, context):
        res = self.db.remove(request.id)
        if res.delete_count == len(request.id):
            message = "Articles removed successfully"
        else:
            message = "Some articles were not deleted"
        return vector_db_pb2.RemoveArticlesResponse(id = request.id, response_message=message)
    
    def UpdateArticles(self,request, context):
        ids = []
        contents = []
        categories = []
        for article in request.articles:
            ids.append(article.id)
            contents.append(article.content)
            categories.append(article.categories)
        res = self.db.upsert(ids, contents, categories)
        print(res)
        
        return vector_db_pb2.UpdateArticlesResponse(id=request.id)

    
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vector_db_pb2_grpc.add_VectorDBServiceServicer_to_server(VectorDBServiceServicer(), server)
    server.add_insecure_port('0.0.0.0:50051')
    server.start()
    server.wait_for_termination()
    
if __name__ == '__main__':
    serve()