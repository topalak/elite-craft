from pydantic import BaseModel
from langchain_core.messages import BaseMessage



class AgentState(BaseModel):
    context_window : list[BaseMessage]
    query : str
    retrieved : str


