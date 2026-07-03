
import os

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

class SummarySignature(dspy.Signature): 
    dialog: str = dspy.InputField()
    summary: str = dspy.OutputField() 

lm = dspy.LM('openai/gpt-4.1-nano', cache=False) 
dspy.settings.configure(lm=lm)

prog_summarizer = dspy.Predict(SummarySignature) 

print(train_examples[0]['summary']) 
pred = prog_summarizer(dialog=summarizer_train_data[0]['dialog']) 
print(pred) 
