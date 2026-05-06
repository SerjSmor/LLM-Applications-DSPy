import dspy

lm = dspy.LM('openai/gpt-4o')
print(lm("Where is Paris? Answer in one word"))
