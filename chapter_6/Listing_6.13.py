
from dspy.teleprompt import GEPA
from typing import Optional
from dspy.teleprompt.gepa.gepa_utils import DSPyTrace, ScoreWithFeedback 

class SimpleSignature(dspy.Signature): 
   """
   Classify the message. Use a short phrase (1 to 4 words) as the class label for the 
   message.
   """
   message: str      = dspy.InputField()
   intent_label: str = dspy.OutputField()

def metric_func(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Optional[DSPyTrace] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[DSPyTrace] = None,
) -> float | ScoreWithFeedback:
    s_ex = set(gold['intent_label'])
    s_pr = set(pred['intent_label'])
    score = len(s_ex.intersection(s_pr)) / len(s_ex.union(s_pr))
    return dspy.Prediction(score=score, feedback=f"Expected {gold.intent_label}. The 
      score reflects the character-level overlap.") 

simple_prog = dspy.Predict(SimpleSignature) 
lm_5 = dspy.LM(model='gpt-5', temperature=1.0, max_tokens=30000) 
gepa_optimizer = GEPA(metric=metric_func, max_metric_calls=100, reflection_lm=lm_5)
gepa_prog = gepa_optimizer.compile(student=simple_prog, trainset=train_set, 
                                   valset=val_set[:50])
prediction = gepa_prog(message="What time is the next flight to Vancouver?")
print(prediction)
print(gepa_prog.signature.instructions)
