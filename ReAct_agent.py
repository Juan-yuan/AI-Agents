import re

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

MAX_STEPS = 8

model = init_chat_model(
    model="qwen2.5:7b",
    model_provider="ollama",
    base_url="http://localhost:11434/",
    temperature=0.1,
    reasoning=False,
    stop=["\nObservation:", "Observation:"],
)


@tool
def multiply(numbers_str: str) -> str:
    """Calculate the product of two integers.

    Input should be two integers separated by a comma, for example: 100,25
    """
    print(f"Running multiplication: {numbers_str}")
    try:
        a_str, b_str = numbers_str.split(",")
        return str(int(a_str.strip()) * int(b_str.strip()))
    except ValueError:
        return "Invalid input. Use two comma-separated integers, for example: 100,25"


@tool
def search_weather(city: str) -> str:
    """Look up the current weather for a given city."""
    print(f"Looking up weather: {city}")
    city_lower = city.lower()
    if "beijing" in city_lower:
        return "Beijing is sunny today, 25 degrees Celsius."
    if "shanghai" in city_lower:
        return "Shanghai is cloudy with light rain today, 22 degrees Celsius."
    if "sydney" in city_lower:
        return "Sydney is partly cloudy today, 18 degrees Celsius."
    return f"Sorry, I do not have weather information for '{city}'."


tools = [multiply, search_weather]
tool_map = {item.name: item for item in tools}
tool_names = ", ".join(tool_map)
tool_descriptions = "\n".join(
    f"- {item.name}: {item.description}" for item in tools
)

REACT_PROMPT = """You are a helpful AI assistant. You can use these tools:

{tool_descriptions}

Follow the ReAct format strictly. Each turn, output ONLY one of the two blocks below.

If you need a tool:
Thought: your reasoning
Action: one of [{tool_names}]
Action Input: the tool input

If you can answer, or every task is done:
Thought: your reasoning
Final Answer: the final answer

Rules:
- Never invent an Observation. The system will add it after you take an Action.
- If the user asks multiple things, handle them one by one.
- For multiply, Action Input must look like 100,25

Example:
User input: What is 6 times 7?
Thought: I need to multiply 6 and 7.
Action: multiply
Action Input: 6,7
Observation: 42
Thought: I have the product.
Final Answer: 42

User input: {input}
{agent_scratchpad}"""


class State(TypedDict):
    input: str
    scratchpad: str
    last_thought: str
    output: str
    steps: int


def parse_react(text: str) -> dict:
    action_match = re.search(r"Action:\s*([^\n]+)", text, re.IGNORECASE)
    input_match = re.search(r"Action Input:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    final_match = re.search(r"Final Answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)

    if action_match and input_match and "Final Answer:" not in action_match.group(0):
        action = action_match.group(1).strip().strip("[]")
        action_input = input_match.group(1).strip().split("\n")[0].strip().strip("'\"")
        return {"type": "action", "action": action, "action_input": action_input}

    if final_match:
        return {"type": "final", "output": final_match.group(1).strip()}

    return {"type": "error"}


def reason(state: State) -> State:
    prompt = REACT_PROMPT.format(
        tool_descriptions=tool_descriptions,
        tool_names=tool_names,
        input=state["input"],
        agent_scratchpad=state["scratchpad"],
    )
    thought = model.invoke(prompt).content.strip()
    print(thought)
    print()
    state["last_thought"] = thought
    state["scratchpad"] += thought + "\n"
    state["steps"] += 1
    return state


def act(state: State) -> State:
    parsed = parse_react(state["last_thought"])
    if parsed["type"] != "action":
        reminder = (
            "Please follow the ReAct format: "
            "Thought / Action / Action Input, or Thought / Final Answer."
        )
        print(f"Observation: {reminder}\n")
        state["scratchpad"] += f"Observation: {reminder}\n"
        return state

    tool = tool_map.get(parsed["action"])
    if tool is None:
        observation = f"Unknown tool '{parsed['action']}'. Choose one of: {tool_names}"
    else:
        observation = tool.invoke(parsed["action_input"])

    print(f"Observation: {observation}\n")
    state["scratchpad"] += f"Observation: {observation}\n"
    return state


def route(state: State) -> str:
    if state["steps"] >= MAX_STEPS:
        return "end"
    parsed = parse_react(state["last_thought"])
    if parsed["type"] == "final":
        return "end"
    return "act"


def finish(state: State) -> State:
    parsed = parse_react(state["last_thought"])
    if parsed["type"] == "final":
        state["output"] = parsed["output"]
    else:
        state["output"] = state["last_thought"]
    return state


def build_react_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("reason", reason)
    graph_builder.add_node("act", act)
    graph_builder.add_node("finish", finish)
    graph_builder.add_edge(START, "reason")
    graph_builder.add_conditional_edges(
        "reason",
        route,
        {
            "act": "act",
            "end": "finish",
        },
    )
    graph_builder.add_edge("act", "reason")
    graph_builder.add_edge("finish", END)
    return graph_builder.compile()


react_agent = build_react_graph()


def run_react(query: str) -> None:
    print(f"User: {query}\n")
    result = react_agent.invoke(
        {
            "input": query,
            "scratchpad": "",
            "last_thought": "",
            "output": "",
            "steps": 0,
        }
    )
    print(f"Final Answer: {result['output']}")
    print("-" * 30)


if __name__ == "__main__":
    run_react("What's the weather like in Sydney today?")
    run_react("What is 100 times 25?")
    run_react("What is 100 times 25? What's the weather in Shanghai?")
