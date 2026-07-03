
class log(): 
    def __init__(self):
       pass
    def log(self, msg):
       pass

class PredictWithLogging(dspy.Module):
   def __init__(self, signature):
      self.query = dspy.Predict(signature) 

   def forward(self, **kwargs):
      answer = self.query(**kwargs) 
      final_answer = dspy.Prediction() 
      for key in answer.keys(): 
        final_answer[key] = answer[key].lower()
      logger.log(f"Question: {input}, Answer: {answer}, Final Answer: {final_answer}") 
      return final_answer

logger = log()
gpt4omini = dspy.LM(model='gpt-4o-mini', api_key=OPENAI_API_KEY)
dspy.settings.configure(lm=gpt4omini)
prog = PredictWithLogging("question -> answer")
prog(question="What is the capital of Spain") 
