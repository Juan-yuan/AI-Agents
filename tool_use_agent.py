from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool


model = init_chat_model(
    model="qwen2.5:7b",
    model_provider="ollama",
    base_url="http://localhost:11434/",
    temperature=0.1,
    reasoning=False,
)


@tool
def multiply(a: int, b: int) -> int:
    """Calculate the product of two integers."""
    return a * b


@tool
def search_weather(city: str) -> str:
    """Look up the current weather for a given city."""
    city_lower = city.lower()
    if "beijing" in city_lower:
        return "Beijing is sunny today, 25 degrees Celsius."
    if "sydney" in city_lower:
        return "Sydney is cloudy with light rain today, 22 degrees Celsius."
    return f"Sorry, I do not have weather information for '{city}'."


agent = create_agent(
    model=model,
    tools=[multiply, search_weather],
    system_prompt=(
        "You are a capable AI assistant that can use tools to answer questions. "
        "Call a tool when it is needed."
    ),
)


def run_agent_and_print(query: str) -> None:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        }
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    run_agent_and_print("What's the weather like in Sydney today?")
    run_agent_and_print("What is 30 times 5? What's the weather in Sydney?")
