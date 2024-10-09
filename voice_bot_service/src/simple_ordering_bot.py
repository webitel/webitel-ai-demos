from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.utils import select_bm25, select_best_match_in_db2
import os
import requests
import json
import datetime
from langchain_community.llms import VLLMOpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM_API_KEY = os.environ.get("LLM_API_KEY")


class SimpleOrderingBot:
    def __init__(self, order_scheme=None):
        # llm = ChatOpenAI(
        #     model="ft:gpt-4o-mini-2024-07-18:webitel:alyaska3:AC5HwwaD",
        #     temperature=0.2,
        #     api_key=OPENAI_API_KEY,
        # )
        llm = VLLMOpenAI(
            openai_api_key="EMPTY",
            openai_api_base="https://9140-45-12-24-58.ngrok-free.app/v1",
            model_name="lora",
            temperature=0,
        )
        if order_scheme is None:
            raise ValueError(
                "You need to provide order_scheme select one from templates.py"
            )

        # llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.0, api_key=LLM_API_KEY)

        # llm = OllamaLLM(model = "llama3.1",base_url = "https://poor-monkeys-behave.loca.lt/", stream= False, temperature=0.0)

        # Set up a parser + inject instructions into the prompt template.
        parser = JsonOutputParser(pydantic_object=order_scheme)
        prompt_text = open("prompts/alyaska_prompt2.txt", "r").read()
        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        print("PROMPT INSTRUCTIONS", parser.get_format_instructions())

        print("PROMPT TEXT", prompt)

        self.llm_chain = prompt | llm | parser

    def answer(self, question, client, chat_history=[], context={}, addresses=[]):
        """
        Process a question and generate a response for the simple ordering bot.

        Args:
            question (str): The user's question.
            client: (weaviate) The client object for database operations.
            chat_history (list, optional): The chat history. Defaults to an empty list.
            context (dict, optional): The context dictionary. Defaults to an empty dictionary.
            addresses (list, optional): The list of addresses. Defaults to an empty list.

        Returns:
            tuple: A tuple containing the generated answer string and the updated context dictionary.
        """
        # for now it is just mocked constant (need to connect with 1c)
        # all_times = [
        #     "2024-09-12 16:00",
        #     "2024-09-12 18:30",
        #     "2024-09-12 12:00",
        #     "2024-09-13 12:00",
        #     "2024-09-12 10:00",
        #     "2024-09-15 12:00",
        #     "2024-09-16 11:30",
        #     "2024-09-16 13:00",
        #     "2024-09-18 11:00",
        #     "2024-09-18 18:00",
        #     "2024-09-19 10:00",
        #     "2024-09-21 12:30",
        #     "2024-09-21 12:30",
        #     "2024-09-23 12:30",
        #     "2024-09-24 10:00",
        #     "2024-09-25 15:00",
        #     "2024-09-26 14:00",
        #     "2024-09-27 12:00",
        #     "2024-09-27 16:00",
        #     "2024-09-28 15:30",
        # ]

        current_date = (
            datetime.datetime.now().isoformat()
        )  # "2024-09-11 13:24"  #   # Get the current date and time in ISO format

        chat_history_str = ""
        if chat_history:
            print("Chat history here", chat_history)
            for item in chat_history:
                chat_history_str += f"{item[1]}: {item[0]}\n"
            print(chat_history_str)

        NER_URL = "http://ner-service:8000/predict_entities"
        labels = ["products"]
        response = requests.post(NER_URL, json={"text": question, "labels": labels})
        if response.status_code == 200:
            entities = response.json()
            unconfirmed_products = [entity["text"] for entity in entities]

            if unconfirmed_products:
                print("Unconfirmed products", unconfirmed_products)
                best_matches = select_best_match_in_db2(client, unconfirmed_products)

                print("Best matches", best_matches)
                best_matches_str = "Ось товари, які я знайшла у нашому каталозі на основі вашого запиту. Я оберу найбільш відповідний продукт:\n"
                catalogue_subset = set(
                    value for sublist in best_matches.values() for value in sublist
                )

                for name, price in catalogue_subset:
                    best_matches_str += f"- {name} за ціною {price}\n"  # Using bullet points for better readability
            else:
                best_matches_str = ""
        else:
            print("Error:", response.text)

        bot_answer = self.llm_chain.invoke(
            {
                "input": question,
                "context": context,
                "chat_history": chat_history_str,
                "date": current_date,  # Pass the current date
                "best_mathces_str": best_matches_str,
            }
        )
        prompt = self.llm_chain.get_prompts()[0]
        print(bot_answer)

        input_prompt = prompt.format(
            input=question,
            context=context,
            chat_history=chat_history_str,
            date=current_date,
            best_mathces_str=best_matches_str,
        )
        data = {"prompt": input_prompt, "gpt_answer": bot_answer}
        # Define the file path
        file_path = "training_data.json"

        # Load existing data if the file exists
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    # Load the existing data
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    # If the file is empty or corrupt, start with an empty list
                    existing_data = []
        else:
            existing_data = []

        # Append the new data
        existing_data.append(data)

        # Save back to the JSON file
        with open(file_path, "w") as f:
            json.dump(existing_data, f, indent=4)  # Pretty print with indent

        # request llama 8b model (0.5) 1-2

        general_answer = bot_answer.get("general_answer")
        # product = bot_answer["current_ordered_products"]
        address = bot_answer.get("address")
        delivery_time = bot_answer.get("delivery_time")
        return_to_main_menu = bot_answer.get("return_to_main_menu")
        # what_i_ordered_last_time = bot_answer.get("what_i_ordered_last_time")
        unconfirmed_products = bot_answer.get("unconfirmed_products")
        confirmed_products = bot_answer.get("confirmed_products")
        print(bot_answer)

        # if search_in_db and unconfirmed_products: #openai
        #     print("Search in db and rerun generation")
        #     unconfirmed_products_names = [x[0] for x in unconfirmed_products]
        #     best_matches = select_best_match_in_db2(client,unconfirmed_products_names)
        #     #dict of unconfirmed to matched products
        #     best_mathces_str = "Ось товари, які я знайшла у нашому каталозі на основі вашого запиту.Я оберу найбільш відповідний продукт:\n"
        #     catalogue_subset = set(value for sublist in best_matches.values() for value in sublist)
        #     for y in catalogue_subset:
        #         best_mathces_str += f"- {y}\n"  # Using bullet points for better readability

        #     print(best_mathces_str)
        #     bot_answer = self.llm_chain.invoke(
        #         {
        #             "input": question,
        #             "context": context,
        #             "chat_history": chat_history_str,
        #             "date": current_date,  # Pass the current date
        #             "best_mathces_str": best_mathces_str
        #         }
        #     )
        #     print(bot_answer)
        #     general_answer = bot_answer.get("general_answer")

        # ask_for_confirmation = bot_answer["ask_for_confirmation"]
        # check if there are new products added to the context by LLM and remove thme from the context
        # def update_products(product, prev_context):
        #     # Extract previous confirmed products for comparison
        #     prev_current_ordered_products = prev_context.get('current_ordered_products', None)
        #     prev_pending_products_for_confirmation = prev_context.get('pending_products_for_confirmation', None)

        #     # If either product or previous confirmed products are None, return an empty list
        #     if product is None or prev_current_ordered_products is None:
        #         return []

        #     # Create a dictionary for previous product quantities
        #     prev_product_quantities = {item[0]: item[1] for item in prev_current_ordered_products}
        #     for item in prev_pending_products_for_confirmation:
        #         if item[0] not in prev_product_quantities:
        #             prev_product_quantities[item[0]] = item[1]
        #     # Initialize an updated product list
        #     updated_product = []

        #     for item in product:
        #         product_name = item[0]
        #         # Check if the product exists in the previous context
        #         if product_name in prev_product_quantities or product_name in prev_pending_products_for_confirmation:
        #             # Use the previous quantity
        #             updated_product.append((product_name, prev_product_quantities[product_name]))

        #     return updated_product

        # # Example usage
        # product = bot_answer["current_ordered_products"]

        # Now `updated_product` contains only the confirmed items with previous quantities
        # print("Prev context", prev_context)
        # product = update_products(product, prev_context)
        # print("Updated product", product)

        # Now `filtered_product` contains only the items that were previously confirmed
        print(bot_answer)
        # product = prev_context.get("confirmed_products", [])

        context = {
            "confirmed_products": bot_answer.get("confirmed_products", []),
            "unconfirmed_products": bot_answer.get("unconfirmed_products", []),
            "address": bot_answer.get("address"),
            "delivery_time": bot_answer.get("delivery_time"),
            "end_conversation": bot_answer.get("end_conversation"),
        }
        # context = {
        #     "confirmed_products": product if product else [],
        #     "address": address,
        #     "delivery_time": delivery_time,
        #     "pending_products_for_confirmation": pending_products_for_confirmation,
        # }
        # print(context["current_ordered_products"])

        ##

        # if not is_request_clear:
        #     return """Вибачте, але я не зрозумів вас. Будь ласка, повторіть.""", context

        # if add_not_specified_products:
        #     return """Вибачте, але я не зрозумів які саме товари ви хочете додати. Будь ласка, уточніть.""", context

        if return_to_main_menu and not (confirmed_products or address or delivery_time):
            return (
                """Натисніть клавішу один щоб зробити нове замовлення, два щоб повторити попереднє замовлення, три щоб дізнатися про поточні знижки, чотири щоб з'єднатийся з оператором""",
                context,
            )

        # if what_i_ordered_last_time:
        #     # probably we can add here products to the context, not sure
        #     return """Звичайно, одну хвилинку перегляну... Іване, четвертого вересня ви замовляли Вода Аляска 18,9 л 4 штуки за ціною 210 грн, Чай Lovare Golden Ceylon (50 пак) 1 штуку за ціною 92.70 грн та Чай Lovare Цитрусова Меліса (24) пак за ціною 65.40 гривень. Бажаєте повторити це замовлення?""", context

        # end separate functions

        # answer = ""
        ##adding confirmed products
        # added = False
        # if confirmed_previous_order:
        #     answer += "Додаю товари з вашого минулого замовлення. \n"

        #     # List of items to add from the previous order
        #     previous_order_items = [
        #         ("Вода Аляска 18,9 л", 4),
        #         ("Чай Lovare Golden Ceylon (50 пак)", 1),
        #         ("Чай Lovare Цитрусова Меліса (24 пак)", 1),
        #     ]

        #     # Get the names of currently added products
        #     current_product_names = {product for product, _ in context["current_ordered_products"]}
        #     # Add previous order items if they haven't been added yet
        #     for product, amount in previous_order_items:
        #         if product not in current_product_names:
        #             added = True
        #             context["current_ordered_products"].append((product, amount))

        #     # Update the answer based on what was added
        #     if previous_order_items:
        #         answer += "Додано:"
        #         for product, amount in previous_order_items:
        #             if product not in current_product_names:
        #                 answer += f"{amount} шт {product}\n"
        #     else:
        #         answer = ""

        # if not added:
        #     answer = ""

        # for item,quantity in context["current_ordered_products"]:
        #     if context.get("pending_products_for_confirmation") and (item,quantity) in context["pending_products_for_confirmation"]:
        #         context["pending_products_for_confirmation"].remove((item,quantity))
        #     if context.get("pending_products_for_confirmation") and [item,quantity] in context["pending_products_for_confirmation"]:
        #         context["pending_products_for_confirmation"].remove([item,quantity])

        # if products_addition_confirmed and context.get("pending_products_for_confirmation"):
        #     print("1")
        #     # Ensure context["products"] is initialized if None or empty
        #     if context.get("current_ordered_products") is None:
        #         context["current_ordered_products"] = []

        #     # Get the names of currently confirmed products
        #     current_current_ordered_products_names = [product for product, _ in context["current_ordered_products"]]
        #     # Initialize a list to track new products added
        #     new_products = []
        #     pending_names = [x for x,y in context["pending_products_for_confirmation"]]
        #     pending_names = select_best_match_in_db(client,pending_names)
        #     # Check and add products from pending_products_for_confirmation
        #     print("2")
        #     for i, (product, amount) in enumerate(context["pending_products_for_confirmation"]) :
        #         if pending_names[i] not in current_current_ordered_products_names:
        #             print("3")
        #             context["current_ordered_products"].append((pending_names[i], amount))
        #             new_products.append((pending_names[i], amount))
        #     print("4")
        #     context["pending_products_for_confirmation"] = []

        #     # Prepare the response message
        #     if new_products:
        #         answer += "Також додаю:"
        #         for product, amount in new_products:
        #             answer += f"{amount} шт {product} за ціною 100 грн,"
        #         answer += "до вашого замовлення. Чи потрібно ще щось додати?"
        #     else:
        #         answer = "Вибачте, я не зовсім вас зрозумів."

        #     return answer, context

        # if pending_products_for_confirmation: #and ask_for_confirmation:
        #     requested_products = [x[0] for x in pending_products_for_confirmation]
        #     final_products = select_best_match_in_db(client, requested_products)
        #     answer += "Ви хочете додати до замовлення: "
        #     for i, final_product in enumerate(final_products):
        #         answer += f"{final_product} в кількості {pending_products_for_confirmation[i][1]} штук "
        #     answer += "?"
        #     return answer, context

        if context["address"]:
            selected_address = select_bm25([context["address"].lower()], addresses)[0]
            context["address"] = selected_address

        # if context["address"] and context["confirmed_products"] and context["delivery_time"]:
        #     # Database search
        #     print("Product db search", context["confirmed_products"])
        #     requested_products = [x[0] for x in context["confirmed_products"]]
        #     final_products = select_best_match_in_db(client, requested_products)
        #     # select_best_address = select_best_match_in_db(client, [address], index_name="Addresses")[0]

        #     # BM25 search
        #     selected_address = select_bm25([context["address"].lower()], addresses)[0]

        #     # Time selection
        #     selected_time_str,selected_time = date_selector(context["delivery_time"], all_times)
        #     order_string = (
        #         "Підкажіть, будь ласка, чи правильно ми зафіксували ваше замовлення: "
        #     )

        #     print("Almost there, final products: ", final_products)
        #     for i, final_product in enumerate(final_products):
        #         order_string += f"{final_product} в кількості {context['confirmed_products'][i][1]} штук "
        #     print("Added prods: ", selected_address)
        #     order_string += f"за адресою {selected_address}, {selected_time_str}?"
        #     print("Added addres: ", selected_address)
        #     context = {
        #         "confirmed_products": final_products,
        #         "address": selected_address,
        #         "delivery_time":  selected_time.strftime("%Y-%m-%d %H:%M:%S"),
        #     }

        #     answer_str = order_string
        # else:

        answer_str = general_answer

        print("Answer", answer_str)
        return answer_str, context
