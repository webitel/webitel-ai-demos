import gradio as gr
import pandas as pd
import random
import json

from automatic_evaluator import Evaluator

evaluator = Evaluator(
    user_metadata={
        "ContractNumber": "№12345",
        "DueDate": "2024-05-13",
        "StartDate": "2024-01-13",
        "TotalDebt": "14.000 грн",
        "credit_body": "10.000 грн",
        "restructurization_paycheck": "300 грн",
        "restructurization_payments_payed": "2",
        "total_restructurization_payments": "24",
        "ApplicationStatus (статус заявки)": "Прийнято на розгляд, відповідь буде надіслана на пошту",
        "PromocodeStatus": "Промокод відправлений на SMS",  # "На жаль, зараз немає діючого промокоду на знижку.\\nРішення про надання знижки приймається автоматично системою.\\nСлідкуйте за СМС повідомленнями, можливо саме Вам надійде наступний промокод на отримання кредиту зі знижкою.",
        "Auth Code (Код для входу)": "Відправлений",
        "user_age": "22 years",
    }
)

# data_path="../data/raw/conversations_6m.json"
# def upload_json(file):
#     evaluator.load_file(file)


def process_history(chat_history):
    res = []
    for i, message in enumerate(chat_history):
        if 2 * i >= len(chat_history):
            print("break")
            break
        message_1 = f'{chat_history[2*i]["sender"]}:\n' + chat_history[2 * i]["message"]
        try:
            message_2 = (
                f'{chat_history[2*i+1]["sender"]}:\n'
                + chat_history[2 * i + 1]["message"]
            )
        except IndexError:
            message_2 = ""
        res.append((message_1, message_2))
    return res


results_of_evalutaion = {}
# Load the chat histories from the data source
model_A_history = json.load(open("../data/outputs/my_results.json"))
model_B_history = json.load(open("../data/outputs/my_results.json"))


assert len(model_A_history) == len(
    model_B_history
), "The number of chat histories should be the same for both models."

all_indicies = list(range(len(model_A_history)))
used_indicies = [0]


def save_choice(choice):
    global used_indicies, chatbot_a, chatbot_b
    results_of_evalutaion[used_indicies[-1]] = choice
    pd.DataFrame(
        results_of_evalutaion.items(), columns=["Chat History Index", "Choice"]
    ).to_csv("evaluation_results.csv", index=False)

    if len(all_indicies) == len(used_indicies):
        # print("All chat histories have been evaluated.")
        gr.Info("All chat histories have been evaluated.")
        return [[("", "")], [("", "")]]

    index_of_chat_history = random.choice(
        list(set(all_indicies).difference(used_indicies))
    )
    used_indicies.append(index_of_chat_history)
    print(model_A_history[index_of_chat_history])
    chatbot_a_chat_history = process_history(model_A_history[index_of_chat_history])
    chatbot_b_chat_history = process_history(model_B_history[index_of_chat_history])

    return [chatbot_a_chat_history, chatbot_b_chat_history]


with gr.Blocks() as demo:
    with gr.Tab("A/B Evaluation"):
        with gr.Row():
            chatbot_a = gr.Chatbot(process_history(model_A_history[0]), label="LLM A")
            chatbot_b = gr.Chatbot(process_history(model_B_history[0]), label="LLM B")

        feedback_text = gr.Textbox(label="Feedback", interactive=False)
        with gr.Row():
            btn_left_better = gr.Button("Left Better")
            btn_right_better = gr.Button("Right Better")
            btn_both_bad = gr.Button("Both Bad")
            btn_both_good = gr.Button("Both Good")

        btn_left_better.click(
            lambda: save_choice("A Better"), inputs=[], outputs=[chatbot_a, chatbot_b]
        )
        btn_right_better.click(
            lambda: save_choice("B Better"), inputs=[], outputs=[chatbot_a, chatbot_b]
        )
        btn_both_bad.click(
            lambda: save_choice("Both Bad"), inputs=[], outputs=[chatbot_a, chatbot_b]
        )
        btn_both_good.click(
            lambda: save_choice("Both Good"), inputs=[], outputs=[chatbot_a, chatbot_b]
        )

    with gr.Tab("Automated Evaluation"):
        eval_btn = gr.Button("Evaluate")
        with gr.Row():
            score_plot = gr.Plot(label="Score Plot")
            exact_match_plot = gr.Plot(label="EM Plot")
            embedding_plot = gr.Plot(label="Embedding Plot")

        eval_btn.click(
            evaluator.evaluate, outputs=[score_plot, exact_match_plot, embedding_plot]
        )
        u = gr.UploadButton(
            "Upload JSON with conversations", file_count="single", type="binary"
        )
        u.upload(evaluator.load_evaluation_data, [u])

        # gr.Textbox("Evaluation Results", text="Evaluation Results will be displayed here.", interactive=False)

demo.launch()
