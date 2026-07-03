
import dspy
def prec_rec_similarity(example: dspy.Example, prediction: dspy.Prediction, trace=None):
    set1 = set(example.summary.split())
    set2 = set(prediction.summary.split())
    intersection = set1.intersection(set2)
    precision = len(intersection) / len(set2)
    recall = len(intersection) / len(set1)
    return min(precision, recall)
