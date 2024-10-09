from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Tuple
from datetime import datetime


class SimpleOrder(BaseModel):
    chain_of_thought: str = Field(
        description=(
            """
When analyzing user intent in messages, follow the patterns established in the provided examples. It is crucial to address all variables outlined in the instructions, even those not explicitly mentioned in the examples. Below is the complete list of variables from the instructions for your reference.
Here is full list of variables in Instructions pending_products_for_confirmation (List[Tuple[str, int]]) Unconfirmed product list, ask_for_confirmation (bool) Request confirmation for new products, products_addition_confirmed (bool) Client approved product addition, delivery_time (datetime) Scheduled delivery time, general_answer (str) Overall response to query, return_to_main_menu (bool) Return to main menu,   add_not_specified_products (bool) Add unspecified products, is_request_clear (bool) Clarity of user input, address (str) Delivery address
In Instructions you can see more detailed description of each variable.

Example :
Клієнт сказав "так, давайте і додайте 2 одиниці товару яблука" у відповідь на "чи хочете ви повторити замовлення?". Я ідентифікую параметри з Instructions, які потрібно змінити тут: pending_products_for_confirmation. Змінна pending_products_for_confirmation повинна бути [['яблука', 2]] та містити товари з минулого замовлення користувача (prev_order), оскільки користувач хоче його повторити та додати до нього товар.

Example :
Клієнт сказав "давайте тільки 2 одиниці товару яблука" у відповідь на "чи хочете ви повторити замовлення?". Я ідентифікую параметри з Instructions, які потрібно змінити тут: pending_products_for_confirmation. Змінна змінна pending_products_for_confirmation повинна бути [['яблука', 2]], prev_order не потрібно додавати, оскільки користувач не сказав, що хоче додати чи повторити товар з минулого замовлення.

Example :
Клієнт сказав "ні, дякую" у відповідь на "чи потрібно ще щось додати?". Я ідентифікую параметри з Instructions, які потрібно змінити тут: pending_products_for_confirmation, products_addition_confirmed, ask_for_confirmation. Змінна pending_products_for_confirmation повинна бути [], products_addition_confirmed повинна бути False, ask_for_confirmation повинна бути False.

Example :
Клієнт сказав "Так, я хотів би дізнатися, що я замовляв минулого разу" у відповідь на "Чи можу я вам чимось допомогти?". Я ідентифікую параметри з Instructions, які потрібно змінити тут: не потрібно змінювати pending_products_for_confirmation. Мені потрібно сказати що клієнт замовляв минулого разу (prev_order) та запитати чи він хоче його повторити.

Example :
Клієнт сказав "Так, 5 штук" у відповідь на "ви хочете додати до замовлення: яблука в кількості 5 штук?". Я ідентифікую параметри з Instructions, які потрібно змінити тут: pending_products_for_confirmation, products_addition_confirmed. Змінна pending_products_for_confirmation повинна бути [['яблука', 5]], а products_addition_confirmed повинна бути True.

Example :
Клієнт сказав "Так, повторіть моє минуле замовлення" у відповідь на "Чи хочете ви повторити своє попереднє замовлення?". Я ідентифікую параметри з Instructions, які потрібно змінити тут: pending_products_for_confirmation. Змінна pending_products_for_confirmation повинна містити минуле замовлення користувача.

Example :
Клієнт сказав "А що я тоді замовляв?" у відповідь на "Чи хочете ви повторити своє попереднє замовлення?". Я ідентифікую параметри з Instructions, які потрібно змінити тут: не потрібно змінювати pending_products_for_confirmation. Мені потрібно сказати що клієнт замовляв минулого разу (prev_order) та запитати чи він хоче його повторити.

Example :
Клієнт сказав "Ні, я нічого не хочу змінювати" у відповідь на "Чи потрібно щось ще додати?". Я ідентифікую параметри з Instructions, які потрібно змінити тут: ask_for_confirmation, products_addition_confirmed, pending_products_for_confirmation. Змінна ask_for_confirmation повинна бути False, products_addition_confirmed повинна бути False, а pending_products_for_confirmation повинна бути [].

Example :
Клієнт сказав "Так" у відповідь на "Ви хочете додати до замовлення: яблука в кількості 4 штук ?". Я ідентифікую параметри з Instructions, які потрібно змінити тут: products_addition_confirmed, pending_products_for_confirmation. Змінна products_addition_confirmed повинна бути True,  pending_products_for_confirmation повинна бути [["яблука",4]].

You should change variable value only if it was mentioned in chain_of_thought. If variable was not mentioned in chain_of_thought, you should not change it.
"""
        ),
    )

    pending_products_for_confirmation: List[Tuple[str, int]] = Field(
        description=(
            "A list of tuples for products and their quantities that have not been confirmed by the client. "
            "These products need to be confirmed with the user before being added to the current order. "
            "If confirmed, we will use products_addition_confirmed to add them to the current order. "
            "Do not add them to the current_ordered_products list on your own. "
            "Remove products from this list if they are confirmed or if the client decides not to add them."
        )
    )
    #    current_ordered_products: List[tuple[str, int]] = Field(
    #         description=("The list of tuples with product and amount that are in current order they should be confirmed by client explicitly",
    #                      "This information is used to track the client's selections and ensure accurate processing of their order.",
    #                      "You may not add products here at all, all you can do is only change the quantity if clients asks to do so.")

    #     )

    ask_for_confirmation: bool = Field(
        description="Whether to ask for confirmation about products in the pending_products_for_confirmation list. This should only be asked for new products."
    )

    products_addition_confirmed: bool = Field(
        description=(
            "Indicates whether the client has approved the addition of products from the 'pending_products_for_confirmation' list to their current order. "
            "Set this flag to True when the client confirms their selection, regardless of the quantity. "
            "After the products are added to 'current_ordered_products', reset this flag to False for the next operation. "
            "This flag is intended for a single confirmation process."
        )
    )

    delivery_time: datetime = Field(
        description="The scheduled time for the order delivery."
    )

    general_answer: str = Field(description="The general answer to the user's query.")

    return_to_main_menu: bool = Field(
        default=False, description="Whether to return to the main menu."
    )

    # confirmed_previous_order: bool = Field(
    #     description=(
    #         "Whether to add the previous order to the list of confirmed products. "
    #         "True if the client asks, tells, or confirms to add or repeat the previous order; false otherwise. "
    #     )
    # )

    # what_i_ordered_last_time: bool = Field(
    #     description="True if you need to tell the client what he ordered last time."
    # )

    add_not_specified_products: bool = Field(
        description=(
            "Indicates whether the client has requested to add products without specifying exact names. "
            "For example, the client might say 'додайте акційні товари' (add promotional items) without providing specific product names."
        )
    )

    is_request_clear: bool = Field(
        description=(
            "Indicates whether the user's input, as recognized by text-to-speech (TTS), was clear and accurately understood. "
            "Clear input is essential for effective interaction. A value of true indicates that the input was clearly understood, "
            "while false suggests potential ambiguity, which may lead to misunderstandings in processing the user's request."
        )
    )

    address: str = Field(
        description="The address to which the order should be delivered."
    )


class SimpleOrder2(BaseModel):
    thoughts: str = Field(
        description="Your thougts before answering. Think step-by-step what you need to do."
    )
    confirmed_products: List[Tuple[str, int, float]] = Field(
        description="The list of products, their quantities and price of item per 1 quantity in the order."
    )
    unconfirmed_products: List[Tuple[str, int, float]] = Field(
        description="The list of products, their quantities and price of item per 1 quantity that have not been confirmed by the client. Add products exactly as client spelled them with full name."
    )

    delivery_time: datetime
    address: str = Field(
        description="The client's address to which the order should be delivered."
    )
    general_answer: str = Field(description="The general answer to the user's query.")
    delivery_time: datetime = Field(
        description="The scheduled time for the order delivery."
    )

    return_to_main_menu: bool = Field(
        default=False, description="Whether to return to the main menu."
    )
    search_in_db: bool = Field(
        description="Whether to search in the database for the product"
    )
    end_conversation: bool = Field(description="Whether to end conversation or not")
