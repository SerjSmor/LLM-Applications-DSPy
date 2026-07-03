
from dspy.evaluate.auto_evaluation import f1_score

class CompleteAndGrounded(dspy.Module):
    def __init__(self, completeness_module:dspy.Module, groundedness_module: dspy.Module):
        self.completeness_module = completeness_module
        self.groundedness_module = groundedness_module
    
    def forward(self, dialog: str, summary: str):
        groundedness = self.groundedness_module(
           dialog=dialog, 
           summary=summary)
        completeness = self.completeness_module(
           dialog=dialog,
           summary=summary)
        return dspy.Prediction(
           score=f1_score(groundedness.groundedness, 
                          completeness.completeness))
