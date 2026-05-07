from dspy.teleprompt import LabeledFewShot

from chapter_5.Listing_5_1 import intent_classifier, lm
from chapter_5.Listing_5_2 import train_set, dev_set

num_examples = 4
labeled_few_shot_optimizer = LabeledFewShot(k=num_examples)
prog_labeled_few_shot = labeled_few_shot_optimizer.compile(student=intent_classifier,
                                                           trainset=train_set)
prediction = prog_labeled_few_shot(**dev_set[0].inputs())
print(prediction)
print(lm.inspect_history(n=1))
print(prog_labeled_few_shot.demos)