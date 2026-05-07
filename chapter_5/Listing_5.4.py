import dspy
from dspy.teleprompt import LabeledFewShot

from chapter_5.Listing_5_1 import intent_classifier, validate_answer
from chapter_5.Listing_5_2 import train_set, val_set

best_score = -1
best_model = None

for num_examples in range(1, 21):
    for _ in range(10):
        labeled_few_shot_optimizer = LabeledFewShot(k=num_examples)
        few_shot_model = labeled_few_shot_optimizer.compile(student=intent_classifier,
                                                            trainset=train_set)

        evaluator = dspy.Evaluate(devset=val_set[:50],
                                  num_threads=5,
                                  display_progress=False,
                                  display_table=False,
                                  provide_traceback=False,
                                  max_errors=0)
        results_part1 = evaluator(few_shot_model, metric=validate_answer)
        if results_part1.score < 80.0:
            continue

        evaluator = dspy.Evaluate(devset=val_set[50:],
                                  num_threads=5,
                                  display_progress=False,
                                  display_table=False,
                                  provide_traceback=False,
                                  max_errors=0)

        results_part2 = evaluator(few_shot_model, metric=validate_answer)
        results = (results_part1.score + results_part2.score * 5) / 6
        if results > best_score:
            best_score = results
            best_model = few_shot_model

print(best_score)
