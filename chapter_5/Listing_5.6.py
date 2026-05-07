from dspy import BootstrapFewShot

from chapter_5.Listing_5_1 import validate_answer, intent_classifier, lm
from chapter_5.Listing_5_2 import train_set, dev_set

optimizer = BootstrapFewShot(
    metric=validate_answer,
    max_bootstrapped_demos=4,
    max_labeled_demos=6,
    max_rounds=10
)

prog_bootrstrap_few_shot = optimizer.compile(student=intent_classifier,
                                             trainset=train_set)
prediction = prog_bootrstrap_few_shot(**dev_set[0].inputs())
print(lm.inspect_history(n=1))
