from src.elite_craft.model_provider import ModelConfig
from config import settings
llm_config = ModelConfig(model='gpt-oss:20b-cloud', api_key=settings.OLLAMA_API_KEY, use_ollama_cloud=True)
llm = llm_config.get_llm()
def main():

    '''from pydantic import BaseModel, Field
    from typing import Literal
    from langchain.agents import create_agent
    from langchain.agents.structured_output import ToolStrategy

    class ProductReview(BaseModel):
        """Analysis of a product review."""
        rating: int | None = Field(description="The rating of the product", ge=1, le=5)
        sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")
        key_points: list[str] = Field(description="The key points of the review. Lowercase, 1-3 words each.")

    agent = create_agent(
        model=llm,
        tools=[],
        response_format=ToolStrategy(ProductReview)
    )

    result = agent.invoke({
        "messages": [{"role": "user",
                      "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
    })
    print(result["structured_response"])

    # ProductReview(rating=5, sentiment='positive', key_points=['fast shipping', 'expensive'])'''

    x = int(input('terim sayisini giriniz'))
    for i in range(x):
        print(i)


if __name__ == '__main__':
    main()



