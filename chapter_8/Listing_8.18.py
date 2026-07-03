
from dspy import Signature, InputField, OutputField

class SummaryCompleteness(Signature):
    """
    Estimate the completeness of a system's responses, against the ground truth summary.
    You will first enumerate key ideas in each of the inputs, discuss their overlap, and 
    then report completeness.
    """
    dialog: str = InputField()
    summary: str = InputField()

    dialog_key_ideas: str = OutputField(desc=
       "enumeration of key ideas in the dialog")
    system_summary_response_key_ideas: str = OutputField(desc=
       "enumeration of key ideas in the system summary response")
    discussion: str = OutputField(desc=
        "discussion of the overlap between ground truth and system response")
    completeness: float = OutputField(desc=
        "fraction (out of 1.0) of ground truth covered by the system response summary")

completeness_predictor = dspy.Predict(SummaryCompleteness)

pred = completeness_predictor(
    dialog=judge_val_data[0].dialog,                              
    summary=judge_val_data[0].summary)    
print(pred)
