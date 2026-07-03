class ChainOfDraft(dspy.Module):
  def __init__(self, signature):
    rationale_field = dspy.OutputField(desc="Think step by step, but only keep \
       a minimum draft for each thinking step, with 5 words at most")
    self.cot = dspy.ChainOfThought(signature, 
                              rationale_field=rationale_field, 
                              rationale_field_type=dspy.OutputField)

  def forward(self, **kwargs):
    return self.cot(**kwargs)    

prog = ChainOfDraft("question -> answer")
prog(question="My aunt is 3 years older than me. I am 40. How old is my aunt")

