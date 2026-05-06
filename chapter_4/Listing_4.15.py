import time

from sklearn.metrics import classification_report
import dspy
import os
from dspy.evaluate import Evaluate

from chapter_3.dspy_structures import IntentSignature
from common.utils import create_examples_from_set, validate_answer

os.environ['OPENAI_API_KEY'] = ""

lm = dspy.LM("openai/gpt-4o-mini", cache=False)
dspy.settings.configure(lm=lm)
classifier = dspy.Predict(IntentSignature)
examples = create_examples_from_set('test')

evaluator = Evaluate(devset=examples, num_threads=8, display_progress=True,
                     display_table=5, provide_traceback=True)
start_time = time.time()
res = evaluator(classifier, metric=validate_answer)

predictions = []
groundtruths = []
for triplett in res.results:
    example, prediction, score = triplett
    groundtruths.append(example.intent_label)
    predictions.append(prediction.intent_label.strip("'"))

end_time = time.time()
print(f"total time: {end_time - start_time}")
print(classification_report(groundtruths, predictions))
