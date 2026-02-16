import dspy
from chapter_9.listing_9_4 import load_data

class FactualAlignmentJudge(dspy.Signature):
    """You are a Factual Alignment Expert. Your job is to evaluate how well 
    an AI response includes the essential information from a ground truth answer 
    (GT answer) according to a given user query.
    
    Note that the Ground Truth (GT Answer), is the "Correct" answer generated 
    by an expert, and was created to evaluate the model, and is NOT part of 
    the AI response or the context.
    
    You will be presented with three elements: a question, a GT answer, and 
    an AI response. Determine how well the AI response includes the essential 
    information from the GT answer that helps to solve the user's query. 
    In case of any additional or extra information present in the AI response, 
    only penalize if it's preventing the user from solving their query.
    
    EVALUATION CRITERIA
5: Complete Match - All essential information
from GT answer appears in AI response, provid-
ing complete solution to the query
4: Strong Match - Most essential information is
present, with only minor details missing that don’t
impact the solution significantly
3: Partial Match - Core information is present but
missing some important details that would help
better solve the query
2: Limited Match - Only basic or partial informa-
tion present, missing several essential elements
needed for the solution
1: Poor Match - Missing most essential informa-
tion or contains incorrect information that could
mislead the user
    
    """
    
    question: str = dspy.InputField(desc="The user's question")
    ground_truth_answer: str = dspy.InputField(desc="Expert-written correct answer")
    ai_response: str = dspy.InputField(desc="AI-generated response to evaluate")
    score: int = dspy.OutputField(
        desc="Score from 1 to 5: 1=no alignment, 2=poor, 3=partial, 4=good, 5=excellent"
    )


class LLMJudge(dspy.Module):
    def __init__(self, predictor):
        self.predictor = predictor

    def forward(self, example: dspy.Example, prediction: dspy.Prediction, trace=None):
        result = self.predictor(
            question=example.question,
            ground_truth_answer=example.answer,
            ai_response=prediction.answer
        )
        score = int(result.score)
        score = max(1, min(5, score))
        return score / 5

if __name__ == "__main__":
    lm = dspy.LM("openai/gpt-4.1-mini", cache=False)
    dspy.configure(lm=lm)

    predictor = dspy.ChainOfThought(FactualAlignmentJudge)
    llm_judge = LLMJudge(predictor)

    train_df, kb_df = load_data()
    first_row = train_df.iloc[0]
    print(f"Question: {first_row['question']}")
    print(f"Ground Truth Answer: {first_row['answer']}")

    example = dspy.Example(
        question="Can I start accepting payments on my site while my Wix Payments account is still under verification?",
        answer="You can start accepting payments on your site using Wix Payments almost immediately. However, we need to verify your identity before your account can be fully activated."
    )

    # Close paraphrase of the ground truth
    good_prediction = dspy.Prediction(
        answer="Yes, you can begin accepting payments with Wix Payments right away. Keep in mind that your identity still needs to be verified for full account activation."
    )
    # Completely wrong answer
    poor_prediction = dspy.Prediction(
        answer="No, you must wait until your Wix Payments account is fully verified before you can accept any payments on your site."
    )

    good_score = llm_judge(example, good_prediction)
    print(f"Good answer score: {good_score:.2f} ({good_score * 100:.0f}%)")

    poor_score = llm_judge(example, poor_prediction)
    print(f"Poor answer score: {poor_score:.2f} ({poor_score * 100:.0f}%)")
