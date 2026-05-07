from dspy import BootstrapFewShotWithRandomSearch

from chapter_5.Listing_5_1 import validate_answer, intent_classifier, lm
from chapter_5.Listing_5_2 import train_set, val_set, dev_set

optimizer = BootstrapFewShotWithRandomSearch(
    metric=validate_answer,
    max_bootstrapped_demos=10,
    max_labeled_demos=5,
    num_threads=10,
    num_candidate_programs=10
)

intent_classifier_bootstrap_few_shot_with_random_search = optimizer.compile(
    intent_classifier,
    trainset=train_set,
    valset=val_set
)

pred = intent_classifier_bootstrap_few_shot_with_random_search(**dev_set[0].inputs())
print(lm.inspect_history(n=1))
