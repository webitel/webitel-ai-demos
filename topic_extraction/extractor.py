from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
import os

api_key = os.getenv("OPENAI_API_KEY")


class Message(BaseModel):
    sender: str
    message: str


class TopicExtractionRequest(BaseModel):
    possible_topics: List[str] = Field(
        ..., description="List of possible topics to choose from"
    )
    messages: List[Message] = Field(
        ..., description="Conversation between user and operator"
    )


class TopicExtractionResponse(BaseModel):
    topics: List[str] = Field(
        ..., description="Single extracted topic from the conversation"
    )
    confidence: float = Field(
        ..., description="Confidence of the extracted topic being correct"
    )


def extract_topics_llm(possible_topics, messages):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key,
    )
    structured_llm = llm.with_structured_output(TopicExtractionResponse)
    prompt = load_prompt()
    topics = "\n"
    for i, topic in enumerate(possible_topics):
        topics += f"Topic {i}.  {topic}\n"
    prompt = prompt.format(possible_topics=topics, messages=messages)
    response = structured_llm.invoke(prompt)
    print(prompt)
    return response


def load_prompt(path="extractor_prompt.txt"):
    with open(path, "r") as file:
        content = file.read()
    return content
