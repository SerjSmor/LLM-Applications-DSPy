class BasicGenerateInstruction(Signature):
    """You are an instruction optimizer for large language models. I will give you a 
    ``signature`` of fields (inputs and outputs) in English. Your task is to propose an 
     instruction that will lead a good language model to perform the task well. Don't be 
     afraid to be creative."""

    basic_instruction = dspy.InputField(
        desc="The initial instructions before optimization")
    proposed_instruction = dspy.OutputField(
        desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = dspy.OutputField(
        desc="The string at the end of the prompt, which will help the model start solving 
              the task",
    )

