
class CompleteAndGrounded(dspy.Module):
    def __init__(self):
        super().__init__()
        self.groundedness_module = dspy.Predict(
           "dialog, summary -> groundedness_score:float")
        self.completeness_module = dspy.Predict(
           "dialog, summary -> completeness_score:float")
        self.score_combiner = dspy.Predict(
           "groundedness_score:float, completeness_score:float -> final_score:float")
    
    def forward(self, dialog: str, summary: str):
        groundedness_score = self.groundedness_module(
           dialog=dialog, 
           summary=summary)
        completeness_score = self.completeness_module(
           dialog=dialog,
           summary=summary)
        return self.score_combiner(groundedness_score=groundedness_score, 
                                   completeness_score=completeness_score)

score_predictor = CompleteAndGrounded()
score_predictor(judge_val_data[0]['dialog'], judge_val_data[0]['summary'])
