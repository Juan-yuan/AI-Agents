from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

MAX_ITERATIONS = 3

model = init_chat_model(
    model="qwen2.5:7b",
    model_provider="ollama",
    base_url="http://localhost:11434/",
    temperature=0.1,
    reasoning=False,
)


class State(TypedDict):
    question: str
    draft: str
    critique: str
    iterations: int


def generate(state: State) -> State:
    if state["iterations"] == 0:
        prompt = (
            "Answer the question clearly and accurately.\n\n"
            f"Question: {state['question']}"
        )
        print("\nGenerating the initial draft...")
    else:
        prompt = (
            "Revise the draft using the critique. Keep what is correct, "
            "fix the problems, and return only the improved answer.\n\n"
            f"Question: {state['question']}\n\n"
            f"Draft:\n{state['draft']}\n\n"
            f"Critique:\n{state['critique']}"
        )
        print(f"\nRevising draft (iteration {state['iterations']})...")

    draft = model.invoke(prompt).content.strip()
    print(f"Draft:\n{draft}")
    state["draft"] = draft
    state["iterations"] += 1
    return state


def reflect(state: State) -> State:
    prompt = (
        "You are a strict critic. Do not rewrite the draft.\n"
        "Check completeness, factual accuracy, and clarity.\n"
        "If the draft needs any improvement, start the reply with REVISE "
        "and list concrete issues.\n"
        "Only if the draft is already excellent, start the reply with APPROVE "
        "and give a one-sentence reason.\n\n"
        f"Question: {state['question']}\n\n"
        f"Draft:\n{state['draft']}"
    )
    print("\nReflecting on the draft...")
    critique = model.invoke(prompt).content.strip()
    print(f"Critique:\n{critique}")
    state["critique"] = critique
    return state


def should_continue(state: State) -> str:
    if state["iterations"] >= MAX_ITERATIONS:
        return "end"
    decision = state["critique"].lstrip().split()[0].upper().strip(":,.")
    if decision == "APPROVE":
        return "end"
    return "revise"


def build_reflection_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("generate", generate)
    graph_builder.add_node("reflect", reflect)
    graph_builder.add_edge(START, "generate")
    graph_builder.add_edge("generate", "reflect")
    graph_builder.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "revise": "generate",
            "end": END,
        },
    )
    return graph_builder.compile()


reflection_agent = build_reflection_graph()


if __name__ == "__main__":
    result = reflection_agent.invoke(
        {
            "question": (
                "Introduce LangChain and explain the difference between "
                "an Agent and a Chain."
            ),
            "draft": "",
            "critique": "",
            "iterations": 0,
        }
    )
    print("\n--- Final answer after reflection ---")
    print(result["draft"])
