import dspy
from dspy.teleprompt import BootstrapFewShot

from chapter_5.Listing_5_1 import validate_answer, ClosedIntentSignature, lm
from chapter_5.Listing_5_2 import train_set, dev_set

gpt4 = dspy.LM(model='gpt-4')

optimizer = BootstrapFewShot(
    metric=validate_answer,
    max_bootstrapped_demos=7,
    max_labeled_demos=5,
    max_rounds=10,
    teacher_settings=dict(lm=gpt4)
)

intent_classifier_cof = dspy.ChainOfThought(ClosedIntentSignature)
bootrstrap_few_shot = optimizer.compile(student=intent_classifier_cof, trainset=train_set)
prediction = bootrstrap_few_shot(**dev_set[0].inputs())
print(lm.inspect_history(n=1))
