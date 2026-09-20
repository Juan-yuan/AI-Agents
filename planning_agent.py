from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
    print(f"Running multiplication: {a} * {b}")
    return a * b


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


planner_prompt = ChatPromptTemplate.from_template(
    """You are a task planner. Break the user's request into a list of
clear, executable steps. Output ONLY the task list, one task per line,
each starting with "- ".

Example:
User task: "Look up the weather in Shanghai, then calculate 20 times 30."
Task list:
- Look up the weather in Shanghai
- Calculate 20 times 30

User task: {user_input}
Task list:
"""
)
planner_chain = planner_prompt | model | StrOutputParser()

executor_agent = create_agent(
    model=model,
    tools=[multiply, search_weather],
    system_prompt=(
        "You are a tool executor. Complete the given task. "
        "Use a tool when it is needed."
    ),
)


def parse_tasks(plan: str) -> list[str]:
    tasks = []
    for line in plan.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.lstrip("-*").strip()
        line = line.lstrip("0123456789.").strip()
        if line and not line.lower().startswith("task list"):
            tasks.append(line)
    return tasks


def run_executor(task: str) -> str:
    result = executor_agent.invoke(
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


def execute_planning_pattern(query: str) -> None:
    print(f"User request: {query}\n")
    print("Creating a plan...")
    plan = planner_chain.invoke({"user_input": query})
    tasks = parse_tasks(plan)

    print("Plan:")
    for i, task in enumerate(tasks, start=1):
        print(f"  {i}. {task}")

    step_results = []
    for i, task in enumerate(tasks, start=1):
        print(f"\n--- Executing task {i}: {task} ---")
        answer = run_executor(task)
        print(answer)
        step_results.append(f"{i}. {task} -> {answer}")

    print("\n--- Final result ---")
    print("\n".join(step_results))


if __name__ == "__main__":
    execute_planning_pattern(
        "First calculate 50 times 60, then tell me the weather in Shanghai."
    )
