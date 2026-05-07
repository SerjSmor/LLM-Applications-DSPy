import dspy
from dspy.teleprompt import BootstrapFewShot

from chapter_5.Listing_5_1 import intent_classifier, lm
from chapter_5.Listing_5_2 import train_set

optimizer = BootstrapFewShot(
    max_bootstrapped_demos=0,
    max_labeled_demos=2,
)

dubious_train_set = [
    dspy.Example(message=x.message, intent_label='XXX').with_inputs('message')
    for x in train_set[:5]]
dubious_train_set.extend(train_set[5:10])

if __name__ == '__main__':
    bootrstrap_few_shot = optimizer.compile(student=intent_classifier,
                                            trainset=dubious_train_set)
    bootrstrap_few_shot(message="What time is the flight to dallas tomorrow?")
    print(lm.inspect_history(n=1))
