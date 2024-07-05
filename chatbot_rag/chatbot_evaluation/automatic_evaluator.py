from langchain.evaluation import load_evaluator
import langchain_core
from langchain_openai import ChatOpenAI
import json
import grpc
import chatbot_pb2
import chatbot_pb2_grpc
from tqdm import tqdm
import pandas as pd
import altair as alt
import requests
from langchain.evaluation import ExactMatchStringEvaluator


class Evaluator:
    def __init__(self, user_metadata):
        self.user_metadata = user_metadata
        self.channel = grpc.insecure_channel("localhost:50056")
        self.stub = chatbot_pb2_grpc.ChatServiceStub(self.channel)
        self.accuracy_criteria = {
            "accuracy": """
You are evaluating the interactions between an assistant and a customer in the banking and credit domains. The evaluation focuses on the accuracy and relevance of the assistant's responses to the customer's queries. Please follow the criteria below to score each response.
Score 1: The answer is completely unrelated to the reference and to the question, or the answer is completely inaccurate.
Score 3: The answer has minor relevance but does not align with the reference.
Score 5: The answer has moderate relevance but contains inaccuracies.
Score 7: The answer aligns with the reference but has minor errors or omissions.
Score 10: The answer is completely accurate and aligns perfectly with the reference.

Important notes:
All questions/inputs are related to banking, credits domain.
The assistant's responses should be evaluated as if they are part of a customer support chat.
Be aware that the reference answer is not always 100% correct and may sometimes contain errors.
"""
        }

        self.evaluator = load_evaluator(
            "labeled_score_string",
            criteria=self.accuracy_criteria,
            llm=ChatOpenAI(model="gpt-4o", temperature=0),
        )
        self.embedding_evaluator = load_evaluator(
            "embedding_distance", embeddings=Embedding()
        )
        self.exact_match_evaluator = ExactMatchStringEvaluator(
            ignore_case=True,
            ignore_numbers=True,
            ignore_punctuation=True,
        )

    def load_evaluation_data(self, data: str | bytes):
        """
        data - (str) path to the data file.
        """
        if isinstance(data, str):
            data = open(data).read()
        elif isinstance(data, bytes):
            data = data.decode("utf-8")
        old = "\\\\"
        new = "\\"
        data_new = data.replace("//", "/")
        data_new = data_new.replace(old, new)
        evaluation_data = json.loads(data_new)[:1]
        print(evaluation_data[0])
        self.evaluation_data = self.process_evaluation_data(evaluation_data)
        print(f"Loaded {len(evaluation_data)} evaluation data.")
        print(self.evaluation_data[0])

        return evaluation_data

    def process_evaluation_data(self, evaluation_data):
        processed_data = []

        # merge messages
        for chat in evaluation_data:
            # chat_id = chat['id']
            messages = chat["messages"]
            processed_data.append(merge_messages(messages))

        return processed_data

    def generate_answer(self, chat_history):
        if len(chat_history) > 6:
            chat_history = chat_history[-6:]
        response = self.stub.Answer(
            chatbot_pb2.MessageRequest(
                user_metadata=self.user_metadata,
                categories=["all"],
                messages=chat_history,
            )
        )
        return response.response_message

    def evaluate_answer(self, user_question, generated_answer, reference_answer):
        # Correct
        eval_result = self.evaluator.evaluate_strings(
            prediction=generated_answer,
            reference=reference_answer,
            input=user_question,
        )
        print(user_question, generated_answer, reference_answer, sep="\n")
        print(eval_result)
        eval_result["exact_match"] = self.exact_match_evaluator.evaluate_strings(
            prediction=generated_answer, reference=reference_answer
        )
        eval_result["embedding_distance"] = self.embedding_evaluator.evaluate_strings(
            prediction=generated_answer, reference=reference_answer
        )
        return eval_result

    def evaluate(self):
        results = []
        for chat in tqdm(self.evaluation_data):
            chat_history = []
            for message in chat:
                if message["operator"] and chat_history:
                    text = self.generate_answer(chat_history)
                    # TODO add eval here
                    evaluation_result = self.evaluate_answer(
                        user_question=chat_history[-1].message,
                        generated_answer=text,
                        reference_answer=message["text"],
                    )

                    results.append(
                        {
                            "user_question": chat_history[-1].message,
                            "generated_answer": text,
                            "reference_answer": message["text"],
                            "score": evaluation_result["score"],
                            "reasoning": evaluation_result["reasoning"],
                            "exact_match": evaluation_result["exact_match"],
                            "embedding_distance": evaluation_result[
                                "embedding_distance"
                            ],
                        }
                    )
                elif message["operator"] and not chat_history:
                    # skip the first message of operator if it is the first message (we will have there greeting message)
                    pass
                else:
                    text = message["text"]
                {"sender": "ai" if message["operator"] else "human", "text": text}
                proto_message = chatbot_pb2.Message(
                    sender="ai" if message["operator"] else "human", message=text
                )
                chat_history.append(proto_message)

        plot_data = []
        exact_match_data = []
        embedding_distance_data = []
        # Plot the results
        for result in results:
            plot_data.append(result["score"])
            exact_match_data.append(result["exact_match"])
            embedding_distance_data.append(result["embedding_distance"])

        pie_chart_scores = create_bar_plot(
            plot_data, title="Distribution of Scores", x_title="Score"
        )
        pie_chart_exact_match = create_bar_plot(
            exact_match_data, title="Distribution of Scores", x_title="Exact Match"
        )
        chart_embedding_scores = (
            alt.Chart(pd.DataFrame({"Embedding Distance": embedding_distance_data}))
            .mark_boxplot()
            .encode(y=alt.Y("Embedding Distance:Q", title="Embedding Distance"))
            .properties(title="Box Plot of Embedding Distances")
        )
        pie_chart_scores.display()
        pie_chart_exact_match.display()
        chart_embedding_scores.display()
        return pie_chart_scores, pie_chart_exact_match, chart_embedding_scores


def create_bar_plot(data, title, x_title):
    df = pd.DataFrame({"Value": data})
    bar_chart = (
        alt.Chart(df)
        .transform_aggregate(count="count()", groupby=["Value"])
        .mark_bar()
        .encode(
            x=alt.X("Value:O", title=x_title),
            y=alt.Y("count:Q", title="Count"),
            tooltip=["Value", "count:Q"],
        )
        .properties(title=title)
    )
    return bar_chart


def merge_messages(data):
    merged_data = []
    current_user = None
    data = [item for item in data if item["name"] != "bot"]
    operator_said_greeting = False
    processed_data = []
    for item in data:
        if operator_said_greeting:
            processed_data.append(item)
        if item["operator"]:
            operator_said_greeting = True

    for item in processed_data:
        if current_user and item["name"] == current_user["name"]:
            current_user["text"] += " " + item["text"]  # Merge texts
        else:
            if current_user:
                merged_data.append(current_user)
            current_user = item

    # Append the last user's messages
    if current_user:
        merged_data.append(current_user)
    return merged_data


class Embedding(langchain_core.embeddings.embeddings.Embeddings):
    def __init__(self):
        pass

    def embed_query(self, text: str):
        response = requests.post(
            "http://localhost:8000/embeddings/", json={"text": text}
        )
        return response.json()["embedding"]

    def embed_documents(self, texts: list[str]):
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings


if __name__ == "__main__":
    embedding_evaluator = load_evaluator("embedding_distance", embeddings=Embedding())
    res = embedding_evaluator.evaluate_strings(
        prediction="Привіт", reference="Добрий день"
    )
    print(res)

    exact_match_evaluator = ExactMatchStringEvaluator(
        ignore_case=True,
        ignore_numbers=True,
        ignore_punctuation=True,
    )

    res = exact_match_evaluator.evaluate_strings(
        prediction="Привіт", reference="Добрий день"
    )
    print(res)
