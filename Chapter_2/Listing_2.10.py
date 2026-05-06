import dspy

dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
lm = dspy.LM('openai/gpt-4o', cache=True, temperature=0.99) 
dspy.configure(lm=lm)
prog = dspy.Predict("question -> answer")
print(prog(question="Compose a 5-line poem about space")) 
print(prog(question="Compose a 5-line poem about space")) 
print(len(lm.history))
