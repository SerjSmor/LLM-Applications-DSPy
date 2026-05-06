import time
import os

import dspy

from chapter_3.dspy_structures import IntentSignature
from common.utils import create_examples_from_set, validate_answer

os.environ['OPENAI_API_KEY'] = ""

def evaluate_baseline(num_examples, model, module_type, num_threads, dataset="test"):
    lm = dspy.LM(model, cache=False)
    dspy.settings.configure(lm=lm)
    if module_type == 'cot':
        classifier = dspy.ChainOfThought(IntentSignature)
    else:
        classifier = dspy.Predict(IntentSignature)
    examples = create_examples_from_set(dataset, n=num_examples)
    evaluate_atis = dspy.Evaluate(devset=examples, num_threads=num_threads,
                                  display_progress=True, display_table=False,
                                  provide_traceback=True, max_errors=100000)
    start_time = time.time()
    overall_score = evaluate_atis(classifier, metric=validate_answer)
    end_time = time.time()
    print(f"total time: {end_time - start_time}")
    return overall_score


for module_type in ['predict', 'cot']:
    for model_name in ['openai/gpt-4o-mini', 'openai/gpt-4.1-nano']:
        for _ in range(3):
            print()
            print(f"Model: {model_name}, Module: {module_type}")
            overall_score = evaluate_baseline(num_examples=876, model=model_name,
                                              module_type=module_type, num_threads=10)
            print(f"Overall score: {overall_score}")
