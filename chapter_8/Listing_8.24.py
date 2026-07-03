
from dspy.teleprompt import SIMBA

def feedback_qa_judge_metric(example: dspy.Example, prediction: dspy.Prediction, 
                             trace=None):    
    feedback = ""
    if (prediction.salient > 0.5) != example.true_score: #A
        feedback += f"""The summary is not good in that the salient field is not correct.    
                    The correct value is {example.true_score}. The LM incorrectly  
                    predicted {prediction.salient}. """ 
    else:
        feedback += f"""The summary is good in that the salient field is correct. The  
                    correct value is {example.true_score} and the LM correctly 
                    predicted {prediction.salient}. """ 
    
    if (prediction.named_entities > 0.5) != example.true_score: #B
        feedback += f"""The summary is not good in that the named_entities field is not   
                    correct. The correct value is {example.true_score}. The LM  
                    incorrectly predicted {prediction.named_entities}. """ 
    else:
        feedback += f"""The summary is good in that the named_entities field is correct.  
                    The correct value is {example.true_score} and the LM correctly  
                    predicted {prediction.named_entities}. """ 
    
    return dspy.Prediction(       
       score=(-1) * \
          abs(example.true_score-((prediction.salient+prediction.named_entities)/2)), 
       feedback=feedback)

simba = SIMBA(metric=feedback_qa_judge_metric, num_candidates=10, max_steps=5, 
              max_demos=10)
simba_qa_judge = simba.compile(student=qa_judge, trainset=judge_val_data)
