from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from datetime import datetime


class SimpleOrder(BaseModel):
    use_previous: bool = Field(description="whether to deliver previous order")
    address: str = Field(description="the address to deliver the order")
    product: List[tuple[str, int]] = Field(
        description="the list of tuples with product and amount"
    )
    delivery_time: datetime = Field(
        description="the scheduled time to deliver the order"
    )
    general_answer: str = Field(description="the general answer to the user query")
    end: bool = Field(
        description="whether to end the conversation with user and to move on to checking whether the order is correct true only if  you know address and product or use_previous is True"
    )
