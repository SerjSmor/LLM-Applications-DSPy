import dspy
import time
import os

from chapter_3.dspy_structures import IntentSignature
from chapter_4.Listing_4_5 import dev_set
from common.utils import validate_answer

os.environ['OPENAI_API_KEY'] = ""


def evaluate_baseline(model="openai/gpt-4o-mini", num_threads=10):
    lm = dspy.LM(model, cache=False)
    dspy.settings.configure(lm=lm)
    classifier = dspy.Predict(IntentSignature)
    evaluator = dspy.Evaluate(devset=dev_set,
                              num_threads=num_threads,
                              display_progress=True,
                              display_table=5,
                              provide_traceback=True,
                              max_errors=100000)
    start_time = time.time()
    results = evaluator(classifier, metric=validate_answer)
    end_time = time.time()
    print(f"total time: {end_time - start_time}")
    return results.score


overall_score = evaluate_baseline(num_threads=10)
