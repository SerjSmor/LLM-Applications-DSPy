
import dspy

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

lm = dspy.LM("openai/gpt-4.1-mini", cache=False)
dspy.configure(lm=lm)

react = dspy.ReAct(
    signature="question -> answer: float",
    tools=[add, subtract, multiply, divide],
    max_iters=4,
)

question = "What is (3 + 5) * 2?"
result = react(question=question)

print(f"Answer: {result.answer}")
print("Trajectory:")
for key, value in result.trajectory.items():
    print(f"  {key}: {value}")
