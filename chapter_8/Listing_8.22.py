
class QASingature(dspy.Signature):
    '''Your task is to judge the summary of a dialog'''

    dialog: str = dspy.InputField()
    summary: str = dspy.InputField()
    
    salient: float = dspy.OutputField(desc="On a scale from 0.0 to 1.0")
    named_entities: float = dspy.OutputField(desc="On a scale from 0.0 to 1.0")
    
def get_final_score(prediction: dspy.Prediction): 
    return ((prediction.salient + prediction.named_entities) / 2) > 0.5  

qa_judge = dspy.ChainOfThought(QASingature)
