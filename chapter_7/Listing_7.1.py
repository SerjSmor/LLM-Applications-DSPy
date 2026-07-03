
import dspy
import os

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
lm = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY)
dspy.settings.configure(lm=lm)
