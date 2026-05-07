from dspy.teleprompt import BootstrapFewShot

from chapter_5.Listing_5_1 import validate_answer, intent_classifier, lm
from chapter_5.Listing_5_7 import dubious_train_set

optimizer = BootstrapFewShot(
    validate_answer,
    max_bootstrapped_demos=3,
    max_labeled_demos=0,
    max_rounds=1
)

bootrstrap_few_shot = optimizer.compile(student=intent_classifier,
                                        trainset=dubious_train_set)
bootrstrap_few_shot(message="What time is the flight to dallas tomorrow?")
print(lm.inspect_history(n=1))
