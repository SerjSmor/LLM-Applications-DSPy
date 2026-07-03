
def qa_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None):
    judge_pred = simba_qa_judge(dialog=example['dialog'], summary=prediction['summary'])
    return get_final_score(judge_pred)  
