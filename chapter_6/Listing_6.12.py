
def feedback_answer(example: dspy.Example, prediction: dspy.Prediction, trace=None): 
    s_ex = set(example['intent_label']) 
    s_pr = set(prediction['intent_label'])
    overlap_score = len(s_ex.intersection(s_pr)) / len(s_ex.union(s_pr)) 
    word_count_score = len(prediction['intent_label'].split()) <= 6
    first_letter_score = prediction['intent_label'][0].isupper()
    last_letter_score = prediction['intent_label'][-1].islower()
    score = overlap_score + word_count_score + first_letter_score + last_letter_score
    feedback=f"Expected {example.intent_label}. The score reflects the character-level 
               overlap. " 
    if word_count_score < 0: 
        feedback += “The labels should be at most 6 words. ”
    if first_letter_score < 0: 
        feedback += “The labels should have the words capitalized. The first letter of the    
                     first word is not. ”
    if last_letter_score < 0: 
        feedback += “The labels should be in lower case other than the first letter of 
                     each word.”
    return dspy.Prediction(score=score, feedback=feedback)
