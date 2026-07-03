
def jaccard_similarity(example: dspy.Example, prediction: dspy.Prediction, trace=None):
    set1 = set(example.summary.split())
    set2 = set(prediction.summary.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) 

jaccard_similarity(summarizer_train_data[0], pred)
