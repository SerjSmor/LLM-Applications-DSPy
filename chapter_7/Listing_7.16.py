
class CriticSignature(dspy.Signature): 
   """
   Evaluate how strong of an answer is given to the question.
   """
   question: str      = dspy.InputField()
   answer: str        = dspy.InputField()
   critique: str      = dspy.OutputField()
   is_extremely_good_answer: bool = dspy.OutputField(desc="Return True only if it's reasonably difficult for a LLM to answer any better. ")

class CriticModule(dspy.Module):
    def __init__(self, max_attemps=3): 
       super().__init__()
       self.max_attempts = max_attempts
       self.task = dspy.Predict("question -> answer") 
       self.critic = dspy.ChainOfThought(CriticSignature) 
       self.prompt_writer = ChainOfDraft(
          "question, answer, previous_instructions, critique -> better_instructions") 

    def forward(self, question):
        attempt_number = 0
        while attempt_number < self.max_attempts: 
          attempt_number += 1
          answer = self.task(question=question) 
          critique = self.critic(question=question, answer=answer) 
          if critique.is_extremely_good_answer: 
              break
          better_instructions = self.prompt_writer( 
              question=question, answer=answer,                                                       
              previous_instructions=self.task.signature.instructions,
              critique=critique).better_instructions
          self.task.signature.instructions = better_instructions
        return answer

lm = dspy.LM("openai/gpt-4o-mini", cache=False)
dspy.settings.configure(lm=lm)
prog = CriticModule() 
prog(question="What is a good way to make money in the import/export business?")
