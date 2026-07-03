
import pandas as pd 
copro = dspy.COPRO(metric=validate_answer, breadth=12, depth=5, track_stats=True, 
                   init_temperature=1.2)
prog_copro2 = copro.compile(intent_classifier, trainset=val_set[:50],
                            eval_kwargs={"num_threads": 2, 'display_progress': True})
print(pd.DataFrame(prog_copro2.results_best[list(prog_copro2.results_best.keys())[0]])) 
