import enum


class IVRChoice(enum.Enum):
    new_order = 0
    repeat_previous_order = 1
    discount = 2
    connect_to_operator = 3
