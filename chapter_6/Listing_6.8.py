
lm_41 = dspy.LM(model='gpt-4.1', temperature=1.0)
mipro2 = dspy.MIPROv2(prompt_model=lm_41, metric=validate_answer, auto="light")
prog_mipro2 = mipro2.compile(intent_classifier, 
                             max_bootstrapped_demos=1, max_labeled_demos=4,    
                             trainset=train_set, valset=val_set)
