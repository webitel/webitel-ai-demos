python -m grpc_tools.protoc -I .   --python_out=. --grpc_python_out=.   chatbot.proto
python -m grpc_tools.protoc -I .   --python_out=. --grpc_python_out=.   vector_db.proto
