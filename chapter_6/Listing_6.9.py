
from dspy.teleprompt import InferRules

infer_rules = InferRules(metric=validate_answer, num_candidates=5, num_rules=10, 
                         num_threads=5)
prog_infer_rules = infer_rules.compile(student=intent_classifier, 
                                       trainset=train_set, valset=val_set)
