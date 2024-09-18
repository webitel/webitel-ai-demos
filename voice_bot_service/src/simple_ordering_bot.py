from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.utils import select_best_match_in_db, select_bm25, date_selector
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM_API_KEY = os.environ.get("LLM_API_KEY")


class SimpleOrderingBot:
    def __init__(self, order_scheme=None):
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            api_key=OPENAI_API_KEY,
        )
        if order_scheme is None:
            raise ValueError(
                "You need to provide order_scheme select one from templates.py"
            )

        # llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.0, api_key=LLM_API_KEY)

        # llm = OllamaLLM(model = "llama3.1",base_url = "https://poor-monkeys-behave.loca.lt/", stream= False, temperature=0.0)

        # Set up a parser + inject instructions into the prompt template.
        parser = JsonOutputParser(pydantic_object=order_scheme)
        prompt = PromptTemplate(
            template="""You are a chatbot for a company Alyaska. You need to help customer make their order. Answer all their questions. You need to ask about address and product they want to order, if the address is said ask about product and wiseversa. You also need to ask about delivery time. Current date is : {date}. Answer only in Ukrainian \n{format_instructions}\n Chat History : {chat_history} \n User input: {input}\n""",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

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
        all_times = [
            "2024-09-12 16:00",
            "2024-09-12 18:30",
            "2024-09-12 12:00",
            "2024-09-13 12:00",
            "2024-09-12 10:00",
            "2024-09-15 12:00",
            "2024-09-16 11:30",
            "2024-09-16 13:00",
            "2024-09-18 11:00",
            "2024-09-18 18:00",
            "2024-09-19 10:00",
            "2024-09-21 12:30",
            "2024-09-21 12:30",
            "2024-09-23 12:30",
            "2024-09-24 10:00",
            "2024-09-25 15:00",
            "2024-09-26 14:00",
            "2024-09-27 12:00",
            "2024-09-27 16:00",
            "2024-09-28 15:30",
        ]

        current_date = "2024-09-11 13:24"  # datetime.now().isoformat()  # Get the current date and time in ISO format

        bot_answer = self.llm_chain.invoke(
            {
                "input": question,
                "context": context,
                "chat_history": chat_history,
                "date": current_date,  # Pass the current date
            }
        )
        general_answer = bot_answer["general_answer"]
        product = bot_answer["product"]
        address = bot_answer["address"]
        delivery_time = bot_answer["delivery_time"]

        if address and product and delivery_time:
            # Database search
            requested_products = [x[0] for x in product]
            final_products = select_best_match_in_db(client, requested_products)
            # select_best_address = select_best_match_in_db(client, [address], index_name="Addresses")[0]

            # BM25 search
            selected_address = select_bm25([address.lower()], addresses)

            # Time selection
            selected_time = date_selector(delivery_time, all_times)

            order_string = (
                "Підкажіть, будь ласка, чи правильно ми зафіксували ваше замовлення: "
            )
            for i, final_product in enumerate(final_products):
                order_string += f"{final_product} в кількості {product[i][1]} штук "
            order_string += f"за адресою {selected_address}, {selected_time}?"

            answer_str = order_string
        else:
            answer_str = general_answer

        context = {
            "product": product,
            "address": address,
            "delivery_time": delivery_time,
        }

        return answer_str, context
