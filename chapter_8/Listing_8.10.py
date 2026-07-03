
from dspy import Evaluate 
from dspy.evaluate.metrics import f1_score

def f1_score_metric(example, prediction, trace=None): 
    return f1_score(example.summary, prediction.summary)

evaluate = Evaluate(devset=summarizer_test_data, num_threads=1, display_table=5,
                    display_progress=True, max_errors=3, provide_traceback=True)
evaluate(prog_summarizer, metric=f1_score_metric)
