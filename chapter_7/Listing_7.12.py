
class PredictWithConciseQuestion(dspy.Module):
    def __init__(self):
        self.query1 = dspy.Predict("question -> concise_question")
        self.query2 = dspy.Predict("question -> answer")

    def forward(self, input: str):
        with dspy.context(lm=gpt4o): 
          concise_question = self.query1(question=input).concise_question
        return self.query2(question=concise_question)

gpt4o = dspy.LM(model='gpt-4o')
gpt4omini = dspy.LM(model='gpt-4o-mini')
dspy.settings.configure(lm=gpt4omini)
prog = PredictWithConciseQuestion()
prog("I'm visiting Spain, a country I enjoy visiting, and would like to know what "   
     "the capital of that country is")
