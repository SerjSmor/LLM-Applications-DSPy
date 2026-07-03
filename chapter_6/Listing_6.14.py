
from dspy.teleprompt import Ensemble

ensemble_optimizer = Ensemble(reduce_fn=dspy.majority)
prog_ensemble = ensemble_optimizer.compile([prog_copro, prog_mipro, prog_infer_rules])
print(prog_ensemble(message="What meals are served on the 3pm flight to Madrid?"))
