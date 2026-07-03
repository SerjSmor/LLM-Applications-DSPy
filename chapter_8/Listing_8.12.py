
def embedding_score(example, prediction, trace=None):
    v1 = encoder_func(example.summary)
    v2 = encoder_func(prediction.summary)
    return cosine_similarity([v1], [v2])[0][0]

embedding_score(summarizer_train_data[0], pred)
