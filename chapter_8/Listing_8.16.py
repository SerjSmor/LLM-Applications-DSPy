from dspy import Signature, InputField, OutputField

class GroundednessSignature(Signature):
    """
    Estimate the groundedness of a system's summary response of a dialog.
    You will first enumerate whatever non-trivial or check-worthy claims are made in the 
    system summary response, and then discuss the extent to which they can be found in the 
    dialog.Ensure that the entities mentioned in the summary are present in the dialog.
    """
    dialog: str = InputField()
    summary: str = InputField()
    
    summary_response_claims: str = OutputField(desc=
       "enumeration of non-trivial or check-worthy claims in the system summary response")
    discussion: str = OutputField(desc=
       "discussion of how supported the claims are by the dialog. Be careful before " 
       "concluding the claims are supported.")
    groundedness: float = OutputField(desc=
       "fraction (out of 1.0) of system summary response fully supported by the dialog."       
       "Score 0.0 if none of the claims found can be supported by the dialog. Score 1.0 " 
       "only if all of the claims are supported by the dialog.")

groundness_predictor = dspy.Predict(GroundednessSignature)

print(judge_val_data[0]) 
pred = groundness_predictor(
    dialog=judge_val_data[0].dialog,                              
    summary=judge_val_data[0].summary)    
print(pred)

print(judge_val_data[150]) 
pred = groundness_predictor(
    dialog=judge_val_data[150].dialog,                              
    summary=judge_val_data[150].summary)    
print(pred)
