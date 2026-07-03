
def reduce_func(predictions):
    intent_labels = [p.intent_label for p in predictions]
    num_unique_labels = len(set(intent_labels))
    if num_unique_labels == 1:
        return dspy.Prediction(intent_label=intent_labels[0])
    return dspy.Prediction(intent_label="Uncertain")

ensemble_optimizer = Ensemble(reduce_fn=reduce_func)
prog_ensemble = ensemble_optimizer.compile([prog_copro, prog_mipro, prog_infer_rules])
print(prog_ensemble(message="What meals are served on the 3pm flight to Madrid?"))
