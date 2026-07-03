
copro = dspy.COPRO(metric=validate_answer, breadth=2, depth=2, init_temperature=1.5, 
                   track_stats=True)
prog_copro = copro.compile(intent_classifier, trainset=val_set, 
                           eval_kwargs={"num_threads": 3})
prediction = prog_copro(message="What meals are served on the 3pm flight to Madrid?")
lm.inspect_history(n=1)
