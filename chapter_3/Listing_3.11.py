import dspy

dspy.settings.configure(
   lm=dspy.LM("openai/gpt-4o-mini", cache=False),
   track_usage=True
)
program = dspy.Predict("question -> answer")
prediction = program(question="What is the capital of France?")
print(prediction.get_lm_usage())
