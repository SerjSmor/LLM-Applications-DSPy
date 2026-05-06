import dspy


def validate_answer(example: dspy.Example, prediction: dspy.Prediction, trace=None):
    return example.intent_label == prediction.intent_label
