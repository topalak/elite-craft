from pydantic import BaseModel, Field




#todo we can add more metadata like subheading (validation) and we can filter them in database if query and database corresponds in that subheading we calculate the similarity inside
# of that subset
class Retriever(BaseModel):
    """
    That tool retrieves related chunks from the database.
    """

    source: str = Field(description="Exact framework name, you need to get that information from user's query"
                                    "example: How works LangGraph's streaming? --> source = langgraph") #todo we must convert any generated value into lower case for more robust output
    query: str = Field(description="query that we calculate the similarity score")




