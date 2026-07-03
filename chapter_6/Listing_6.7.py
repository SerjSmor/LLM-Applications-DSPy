
mipro = dspy.MIPROv2(metric=validate_answer, auto="light")
prog_mipro = mipro.compile(intent_classifier, 
                           max_bootstrapped_demos=1, max_labeled_demos=4,    
                           trainset=train_set, valset=val_set)
