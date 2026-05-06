import dspy

lm = dspy.LM('openai/gpt-4o')
print(lm(messages=[
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "Where is Ottawa? Answer in one word"
    },
    {
        "role": "assistant",
        "content": "Ottawa is in Canada"
    },
    {
        "role": "user",
        "content": "Where is Paris? Answer in one word"
    }
]))
