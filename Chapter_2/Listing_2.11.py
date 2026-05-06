import os
import dspy

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

lm_mini = dspy.LM(model='gpt-4o-mini', cache=False, temperature=0.0)
lm_41 = dspy.LM(model='gpt-4.1-turbo', cache=False, temperature=0.0)
dspy.settings.configure(lm=lm_41)
prog = dspy.Predict('question->answer')
prediction = prog(question='Write a 5-line poem about toasters')
print(prediction)

with dspy.context(lm=lm_mini):
    prediction = prog(question='Write a 5-line poem about toasters')
    print(prediction)

prog.set_lm(lm_41)
prediction = prog(question='Write a 5-line poem about toasters')
print(prediction)
