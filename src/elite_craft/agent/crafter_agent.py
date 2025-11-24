from elite_craft.tools.retriever import Retriever


class Crafter:

    def __init__(self,embedding_model_name: str, supabase_url: str, supabase_api_key: str):
        self.retriever = Retriever(supabase_url=supabase_url, supabase_api_key=supabase_api_key, embedding_model_name=embedding_model_name)


    def ask(self, query: str) -> list[dict]:
        response = self.retriever.retrieve_relevant_chunks(query)
        return response

