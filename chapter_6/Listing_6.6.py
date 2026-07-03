from dspy.teleprompt import LabeledFewShot
labeled_few_shot_optimizer = LabeledFewShot(k=4)
prog_few_shot = labeled_few_shot_optimizer.compile(student=intent_classifier,  
                                                   trainset=train_set)
copro = dspy.COPRO(metric=validate_answer, breadth=2, depth=2, track_stats=True)
prog_copro_3 = copro.compile(prog_few_shot, trainset=val_set,
                             eval_kwargs={"num_threads": 3, 'display_progress': True})
