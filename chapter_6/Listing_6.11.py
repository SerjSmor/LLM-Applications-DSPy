
class SimpleSignature(dspy.Signature): 
   """
   Classify the message into an appropriate class.
   """
   message: str      = dspy.InputField()
   intent_label: str = dspy.OutputField()

def feedback_answer(example: dspy.Example, prediction: dspy.Prediction, trace=None): 
    s_ex = set(example['intent_label'])
    s_pr = set(prediction['intent_label'])
    score = len(s_ex.intersection(s_pr)) / len(s_ex.union(s_pr))
    return dspy.Prediction(score = score, feedback=f"Expected {example.intent_label}. The 
        score reflects the character-level overlap.")

def evaluate_prog(prog, num_threads=10): 
    evaluator= dspy.Evaluate(devset=dev_set[:50], 
                             num_threads=num_threads, 
                             display_progress=True, 
                             display_table=50, 
                             provide_traceback=True)
    results = evaluator(prog, metric=feedback_answer) 
    return results.score

prog = dspy.Predict(SimpleSignature) 
overall_score = evaluate_prog(prog, num_threads=10) 
print(f"Overall score: {overall_score}")

lm_41 = dspy.LM(model='gpt-4.1', temperature=1.0)
simba_optimizer = SIMBA(metric=feedback_answer, prompt_model=lm_41, num_candidates=3, 
                        max_steps=3, max_demos=10)
prog_simba_2 = simba_optimizer.compile(student=prog, trainset=train_set) 
overall_score = evaluate_prog(prog_simba_2, num_threads=10) 
print(f"Overall score: {overall_score}")
