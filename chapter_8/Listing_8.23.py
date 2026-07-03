
from dspy import Evaluate 

def qa_judge_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None):
    predicted_score = get_final_score(prediction)
    return example.true_score == predicted_score

evaluate = Evaluate(devset=judge_val_data, num_threads=1, display_table=5,
                    display_progress=True, max_errors=3, provide_traceback=True)

evaluate(qa_judge, metric=qa_judge_metric)
