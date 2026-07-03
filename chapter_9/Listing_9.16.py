import dspy

attempts = [] 

def length_reward(args, pred: dspy.Prediction) -> float: 
    """Score 1.0 if the answer is within the word limit, 0.0 otherwise."""
    MAX_WORDS = 5
    word_count = len(pred.response.split())
    passed = word_count <= MAX_WORDS
    try:
        attempts.append({"words": word_count, "passed": passed, "answer": pred.response})
    except Exception as e:
        print("Error:", e) 
    status = "PASS" if passed else "FAIL"
    print(f"[Attempt {len(attempts)}] {word_count} words (limit {MAX_WORDS}) -> {status}")
    return 1.0 if passed else 0.0

class RefinedReActWithMemory(dspy.Module):
    def __init__(self):
        self.session_history = dspy.History(messages=[])
        self.respond = dspy.ReAct('session_history, question -> response', tools=[]) 
        self.refined_response = dspy.Refine(module=self.respond,  
                                            N=3, 
                                            reward_fn=length_reward, 
                                            threshold=1.0) 

    def forward(self, question):        
        response = self.refined_response(
            session_history=self.session_history, question=question) 
        self.session_history.messages.append({"question": question, **response})
        return response

lm = dspy.LM("openai/gpt-4.1-mini", cache=False)
dspy.configure(lm=lm)

prog = RefinedReActWithMemory()       
print(prog(question = "What is the capital of Spain?"))
