
class SimpleModule(dspy.Module):
   def __init__(self): 
      super().__init__()
      self.query = dspy.Predict("input -> output") 

   def forward(self, input): 
      return self.query(input=input)

prog = SimpleModule() 
prog(input="What is the capital of France?")
