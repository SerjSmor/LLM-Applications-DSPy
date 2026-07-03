
def evaluate_summary(example: dspy.Example, prediction: dspy.Prediction, trace=None):
    judge = dspy.Predict("ground_truth_summary, system_summary -> summary_quality: float")
    return judge(
        ground_truth_summary=example.summary,
        system_summary=prediction.summary)

evaluate_summary(summarizer_train_data[0], pred)
The judge here is a very simple program, based on 
