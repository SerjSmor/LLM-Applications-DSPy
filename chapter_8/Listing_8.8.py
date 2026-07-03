
from dspy.evaluate.metrics import f1_score
print(f1_score(summarizer_train_data[0].summary, pred.summary))
