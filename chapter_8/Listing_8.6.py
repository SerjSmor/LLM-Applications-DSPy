
def jaccard_similarity(example: dspy.Example, prediction: dspy.Prediction, trace=None):
    def get_ngrams(s, n):
        return set([tuple(s[i:i+n]) for i in range(len(s) - n + 1)]) 

    set1 = get_ngrams(example.summary.split(), n=2)
    set2 = get_ngrams(prediction.summary.split(), n=2)

    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

jaccard_similarity(summarizer_train_data[0], pred) 
