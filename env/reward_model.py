from pydantic import BaseModel


class Reward(BaseModel):
    value: float
