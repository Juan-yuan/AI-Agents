import datetime

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

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
    print(f"\n[Math specialist] Running multiplication: {a} * {b}")
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Calculate the sum of two integers."""
    print(f"\n[Math specialist] Running addition: {a} + {b}")
    return a + b


@tool
def search_weather(city: str) -> str:
    """Look up the current weather for a given city."""
    print(f"\n[Info specialist] Looking up weather: {city}")
    city_lower = city.lower()
    if "beijing" in city_lower:
        return "Beijing is sunny today, 25 degrees Celsius."
    if "shanghai" in city_lower:
        return "Shanghai is cloudy with light rain today, 22 degrees Celsius."
    if "sydney" in city_lower:
        return "Sydney is partly cloudy today, 18 degrees Celsius."
    return f"Sorry, I do not have weather information for '{city}'."


@tool
def get_current_date() -> str:
    """Get the current date."""
    print("\n[Info specialist] Getting the current date...")
    return datetime.date.today().strftime("%Y-%m-%d")


math_agent = create_agent(
    model=model,
    tools=[multiply, add],
    system_prompt="You are a math specialist. Use the math tools to answer.",
)

info_agent = create_agent(
    model=model,
    tools=[search_weather, get_current_date],
    system_prompt="You are an information specialist. Use the query tools to answer.",
)


class State(TypedDict):
    query: str
    math_task: str
    info_task: str
    math_result: str
    info_result: str
    final_answer: str


def run_specialist(agent, task: str) -> str:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": task,
                }
            ]
        }
    )
    return result["messages"][-1].content


def math_specialist(state: State) -> dict:
    print("\n[Coordinator] Assigning work to the math specialist...")
    math_result = run_specialist(math_agent, state["math_task"])
    print(f"[Coordinator] Math specialist returned: {math_result}")
    return {"math_result": math_result}


def info_specialist(state: State) -> dict:
    print("\n[Coordinator] Assigning work to the info specialist...")
    info_result = run_specialist(info_agent, state["info_task"])
    print(f"[Coordinator] Info specialist returned: {info_result}")
    return {"info_result": info_result}


def summarize(state: State) -> dict:
    print("\n[Coordinator] Combining the specialist results...")
    prompt = (
        "You are a coordinator. Combine the specialist results into one "
        "complete answer to the user request.\n\n"
        f"User request: {state['query']}\n\n"
        f"Math result: {state['math_result']}\n\n"
        f"Info result: {state['info_result']}"
    )
    final_answer = model.invoke(prompt).content.strip()
    print(f"\nFinal answer:\n{final_answer}")
    return {"final_answer": final_answer}


def build_multi_agent_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("math_specialist", math_specialist)
    graph_builder.add_node("info_specialist", info_specialist)
    graph_builder.add_node("summarize", summarize)
    graph_builder.add_edge(START, "math_specialist")
    graph_builder.add_edge(START, "info_specialist")
    graph_builder.add_edge(["math_specialist", "info_specialist"], "summarize")
    graph_builder.add_edge("summarize", END)
    return graph_builder.compile()


multi_agent = build_multi_agent_graph()


if __name__ == "__main__":
    result = multi_agent.invoke(
        {
            "query": (
                "First calculate 25 times 4, then tell me the weather in "
                "Beijing today and the current date."
            ),
            "math_task": "Calculate 25 times 4",
            "info_task": "Look up today's weather in Beijing and the current date",
            "math_result": "",
            "info_result": "",
            "final_answer": "",
        }
    )
    print("-" * 30)
    print(result["final_answer"])
