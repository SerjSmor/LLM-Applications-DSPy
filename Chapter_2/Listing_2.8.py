import dspy

MISTRAL_API_KEY = ""
lm = dspy.LM("mistral/mistral-tiny", api_key=MISTRAL_API_KEY)
dspy.configure(lm=lm)
prog = dspy.Predict("question -> answer")
prog(question="What is the capital of Russia?")
