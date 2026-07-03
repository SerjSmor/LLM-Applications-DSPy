
import dspy

class HardQuestion(dspy.Signature):
    """
    Answer a question that probably requires writing and executing Python code.
    """
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Accurate response to the question")

def execute_python_code(python_code: str) -> str:
    """Execute Python code and return the output."""
    print(f"{python_code=}")
    s = dspy.PythonInterpreter().execute(python_code) 
    return s

lm = dspy.LM("openai/gpt-4.1-mini", cache=False)
dspy.configure(lm=lm) 

react_agent = dspy.ReAct(
    signature=HardQuestion,
    tools=[execute_python_code],
    max_iters=4,
)

question = "Find the smallest number that is the sum of its digits"
result = react_agent(question=question)
print(result)
