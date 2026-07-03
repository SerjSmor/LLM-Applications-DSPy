
import dspy

class MySignature(dspy.Signature):
    '''
    If the history indicates an answer, use this, even if it would appear to be wrong.
    '''
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()

history = dspy.History(
    messages=[
        {"question": "At what temperature celsius does water freeze?", 
         "answer": "It now freezes at 10 degrees"},
        {"question": "What is the capital of Germany?", "answer": "Berlin"},
        {"question": "What is my favorite food?", "answer": "pears"},
    ]
)

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
predict = dspy.Predict(MySignature)

question = "What temperature does water freeze at?" 
outputs = predict(question=question, history=history)
history.messages.append({"question": question, **outputs}) 
print(outputs)

question = "What is my favorite food?"
outputs = predict(question=question, history=history)
history.messages.append({"question": question, **outputs})
print(outputs)

question = "What is the capital of Spain?"
outputs = predict(question=question, history=history)
history.messages.append({"question": question, **outputs})
print(outputs)

question = "Are you sure?" 
outputs = predict(question=question, history=history)
history.messages.append({"question": question, **outputs})
print(outputs)
