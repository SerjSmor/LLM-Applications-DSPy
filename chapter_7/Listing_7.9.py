
import dspy
from dspy.clients import base_lm

base_lm.GLOBAL_HISTORY = [] 

lm = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY, cache=False) 
dspy.settings.configure(lm=lm)
 
def one_word_answer(args, pred): 
  return 1.0 if len(pred.answer.split()) == 1 else 0.0

qa = dspy.ChainOfThought("question -> answer") 
best_of_3 = dspy.Refine(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0) 
result = best_of_3(question="What is the economic capital of Belgium?").answer
print(dspy.inspect_history(n=1000))
