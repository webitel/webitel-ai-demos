from langchain_openai import ChatOpenAI
import minio
import pandas as pd
from io import BytesIO
from constants import default_reranker_file
import os 
import json
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import logging

openai_model = os.environ.get("OPENAI_MODEL")
device = os.environ.get("DEVICE")
db_host = os.environ.get("HOST")
http_port = int(os.environ.get("HTTP_PORT"))
grpc_port = int(os.environ.get("GRPC_PORT"))
minio_login = os.environ.get('MINIO_ROOT_USER')
minio_password = os.environ.get('MINIO_ROOT_PASSWORD')
minio_bucket_name = os.environ.get('MINIO_DEFAULT_BUCKETS')
minio_url = os.environ.get('MINIO_URL')


def load_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    return prompt


class ParaphrasedQuery(BaseModel):
    """You have performed query expansion to generate a paraphrasing of a question."""

    paraphrased_query: str = Field(
        ...,
        description="A unique paraphrasing of the original question.",
    )

def generate_questions(question, answer):
    llm = ChatOpenAI(model=openai_model, temperature=0.4,)


    template = load_prompt('prompts/generate_similar_questions_prompt.txt')

    prompt = PromptTemplate.from_template(template)
    llm_with_tools = llm.bind_tools([ParaphrasedQuery])
    syntetitc_data_generator = prompt | llm_with_tools
    res = syntetitc_data_generator.invoke({"example": question + " Відповідь: " + answer})
    generated_questions = ""
    for i,tool_call in enumerate(res.additional_kwargs['tool_calls']):
        generated_question = json.loads(tool_call['function']['arguments'])['paraphrased_query']
        generated_questions += f"Q{i}: {generated_question} \n" 
    return generated_questions



def add_sample_for_cross_encoder(df, bucket_name = 'chatbot-rag'):
        # Initialize Minio client
        client = minio.Minio(minio_url,
                            access_key=minio_login,
                            secret_key=minio_password,
                            secure=False)
        try:
        # Read the CSV file from Minio bucket
            csv_data = client.get_object(
                bucket_name, default_reranker_file
            ).data
            df_original = pd.read_csv(BytesIO(csv_data))

            # Concat two dfs 
            df = pd.concat([df_original, df])
        except Exception as e:
            logging.error(f"Some issue occured while fetching samples from Minio!, Error: {e}")
        # Save the updated DataFrame to a new CSV file
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        csv_buffer = BytesIO(csv_bytes)
        
        # Upload the updated CSV file back to Minio bucket
        client.put_object(bucket_name, default_reranker_file, data = csv_buffer,length = len(csv_bytes), content_type='application/csv')
        return "Samples added successfully!"


def get_samples_from_minio(bucket_name = 'chatbot-rag'):
    # Initialize Minio client
    client = minio.Minio(minio_url,
                        access_key=minio_login,
                        secret_key=minio_password,
                        secure=False)
    try:
        # Read the CSV file from Minio bucket
        csv_data = client.get_object(
            bucket_name, default_reranker_file
        ).data
        df = pd.read_csv(BytesIO(csv_data))
        return df
    except Exception as e:
        return f"Some issue occured while fetching samples from Minio!, Error: {e}"

if __name__ == '__main__':
    df = pd.DataFrame([{'query':'ss','positive':'sss','negative':'s'},{'query':'ssss','positive':'sss','negative':'s'}])
    add_sample_for_cross_encoder(df,minio_bucket_name)
