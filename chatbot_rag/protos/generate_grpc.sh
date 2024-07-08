python -m grpc_tools.protoc -I protos --python_out=. --grpc_python_out=. protos/chatbot.proto
python -m grpc_tools.protoc -I protos --python_out=. --grpc_python_out=. protos/vector_db.proto

