lm_5 = dspy.LM(model='gpt-5', temperature=1.0, max_tokens=30000)
copro = dspy.COPRO(prompt_model=lm_5, init_temperature=1.0, metric=validate_answer, 
                   breadth=2, depth=2, track_stats=True)
prog_copro_2 = copro.compile(intent_classifier, trainset=val_set,
                             eval_kwargs={"num_threads": 3, 'display_progress': True})
