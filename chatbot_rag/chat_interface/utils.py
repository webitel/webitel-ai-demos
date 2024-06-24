import minio
from sentence_transformers import CrossEncoder
import os
from constants import reranker_models_path
import shutil 
import logging

def load_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    return prompt


def load_model(model_name,device,bucket_name,access_key,secret_key,minio_url):
    if os.path.exists(os.path.join(reranker_models_path,model_name)):
        return CrossEncoder(model_name, max_length=512, device=device)
    
    client = minio.Minio(minio_url,
                        access_key=access_key,
                        secret_key=secret_key,
                        secure=False)
    
    if os.path.exists(reranker_models_path) and len(os.listdir(reranker_models_path)) > 3:
        ## remove oldest folder
        dirs = [_dir for _dir in os.listdir(reranker_models_path)]
        sorted_dirs = list(sorted(dirs, key=lambda x: os.stat(os.path.join(reranker_models_path,x)).st_mtime))
        print(f"removing {sorted_dirs[-1]}")
        logging.error(f"removing {sorted_dirs[-1]}")
        shutil.rmtree(os.path.join(reranker_models_path,sorted_dirs[-1]))
    
    for item in client.list_objects(bucket_name,prefix = os.path.join(reranker_models_path,model_name) ,recursive=True):
        client.fget_object(bucket_name,item.object_name,item.object_name)
    
    
    return CrossEncoder(os.path.join(reranker_models_path,model_name), max_length=512, device=device)

if __name__ == "__main__":
    # model_name = "DiTy/cross-encoder-russian-msmarco"
    # device = "cuda"
    # minio_url = 'localhost:9002'
    model = load_model("test_model",'cpu','chatbot-rag','minioroot','miniopassword','localhost:9002')
    print(model.predict(["привет"]))