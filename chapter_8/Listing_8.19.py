
def groundedness_and_completeness(example: dspy.Example, prediction: dspy.Prediction, 
                                  trace=None):
    groundedness = groundness_predictor(
        dialog=example.dialog,
        summary=prediction.summary)
    completeness = completeness_predictor(
        dialog=example.dialog,
        summary=prediction.summary)
    return (groundedness.groundedness + completeness.completeness) / 2

pred = groundedness_and_completeness(judge_val_data[0], 
                                     dspy.Prediction(summary=judge_val_data[0].summary))    
print(pred)
