from typing import List

import dspy

from chapter_5.Listing_5_1 import validate_answer
from chapter_5.Listing_5_2 import dev_set


def evaluate_model(examples: List[dspy.Example], classifier: dspy.Module):
    evaluator = dspy.Evaluate(devset=examples, num_threads=5, display_progress=False,
                              display_table=False, provide_traceback=False)
    evaluator(classifier, metric=validate_answer)


if __name__ == '__main__':
    evaluate_model(dev_set, best_model)
