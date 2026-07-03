class GenerateInstructionGivenAttempts(dspy.Signature):
    """You are an instruction optimizer for large language models. I will give some task 
    instructions I've tried, along with their corresponding validation scores. The 
    instructions are arranged in increasing order based on their scores, where higher 
    scores indicate better quality.

    Your task is to propose a new instruction that will lead a good language model to 
    perform the task even better. Don't be afraid to be creative."""

    attempted_instructions = dspy.InputField()
    proposed_instruction = dspy.OutputField(desc="The improved instructions for the 
        language model")
    proposed_prefix_for_output_field = dspy.OutputField(desc="The string at the end of the 
        prompt, which will help the model start solving the task",)
