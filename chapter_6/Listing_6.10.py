
from dspy.teleprompt import SIMBA

simba = SIMBA(metric=validate_answer, num_candidates=3, max_steps=20, max_demos=10)
prog_simba = simba.compile(student=intent_classifier, trainset=train_set)
prediction = prog_simba(message="What meals are served on the 3pm flight to Madrid?")
print(prediction)
print(lm.inspect_history(n=1))
