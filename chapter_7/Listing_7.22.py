from dspy import BootstrapFewShot

class PredictWithClearQuestion(dspy.Module): 
    def __init__(self, signature):
        self.query1 = dspy.Predict("question -> clear_question")
        self.query2 = dspy.Predict(signature)

    def forward(self, **args):
        q = args['message']
        args['message'] = self.query1(question=q).clear_question
        return self.query2(**args)

lm = dspy.LM(model='gpt-4o-mini')
dspy.settings.configure(lm=lm)
prog = PredictWithClearQuestion(ClosedIntentSignature) 

optimizer = BootstrapFewShot( 
    metric=validate_answer,
    max_bootstrapped_demos=3,
    max_labeled_demos=0,
    max_rounds=1
)

prog_bootrstrap_few_shot = optimizer.compile(student=prog, 
                                             trainset=train_set) 
prediction = prog_bootrstrap_few_shot(**dev_set[0].inputs()) 
print(lm.inspect_history(n=2)) 
