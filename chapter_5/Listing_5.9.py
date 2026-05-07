from dspy.teleprompt import BootstrapFewShot

from chapter_5.Listing_5_1 import intent_classifier, lm
from chapter_5.Listing_5_7 import dubious_train_set


def validate_answer_numeric(example, prediction, trace=None):
    s_ex = set(example['intent_label'])
    s_pr = set(prediction['intent_label'])
    score = len(s_ex.intersection(s_pr)) / len(s_ex.union(s_pr))
    if trace is None:
        return score
    else:
        return score > 0.98


optimizer = BootstrapFewShot(
    metric=validate_answer_numeric,
    max_bootstrapped_demos=3,
    max_labeled_demos=0,
    max_rounds=1
)

bootrstrap_few_shot = optimizer.compile(student=intent_classifier,
                                        trainset=dubious_train_set)
bootrstrap_few_shot(message="What time is the flight to dallas tomorrow?")
print(lm.inspect_history(n=1))
