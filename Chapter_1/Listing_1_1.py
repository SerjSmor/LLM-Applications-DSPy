import dspy

OPENAI_API_KEY = ""
lm = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY)
dspy.settings.configure(lm=lm)
predictor = dspy.Predict("question, context -> answer, confidence")
prediction = predictor(question="What is the capital of France?", context="")
print(prediction.answer, prediction.confidence)
