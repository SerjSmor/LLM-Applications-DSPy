import dspy

OPENAI_API_KEY = ""
lm = dspy.LM(model='gpt-4o-mini', api_key=OPENAI_API_KEY)
dspy.settings.configure(lm=lm)
prog = dspy.Predict("question -> answer")
prediction = prog(question="What is the capital of France?")
print(prediction)
