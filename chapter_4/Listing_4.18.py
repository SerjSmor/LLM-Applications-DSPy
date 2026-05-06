from collections import Counter

import dspy

from chapter_3.dspy_structures import IntentSignature
from chapter_4.Listing_4_5 import dev_set
from common.utils import validate_answer

lm = dspy.LM('gpt-4o-mini', cache=False)
dspy.settings.configure(lm=lm)
classifier = dspy.Predict(IntentSignature)
for example in dev_set:
    examples = [example] * 10
    evaluator = dspy.Evaluate(devset=examples, num_threads=10, display_progress=False,
                              display_table=False, provide_traceback=False,
                              max_errors=100000)
    res = evaluator(classifier, metric=validate_answer)
    predictions = [prediction.intent_label for _, prediction, _ in res.results]
    print(predictions)
    print(Counter(predictions))
