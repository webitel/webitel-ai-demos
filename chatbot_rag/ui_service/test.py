import vector_db_pb2_grpc, vector_db_pb2
impot grpc

channel = grpc.insecure_channel('localhost:50053')
stub = vector_db_pb2_grpc.VectorDBServiceStub(channel)    


