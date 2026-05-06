role_string_1 = "You are an expert of customer service intent classification. \n"
role_string_2 = "Act as a person who is able to classify user messages very accurately. \n"
task_string = "Your task is to classify the intent of customer messages of an airline company into one of the provided labels.\n"
usage_string = "Your classification will be used to route the user message to the correct customer service team\n"
input_string = "Input: customer message\n"
output_string = f"Output: One of the following classes: {''.join(unique_intents)}\n"
output_format_string = "The output should be exactly one of the listed labels\n"
stakes_string = "If you get this right, I will give you a million dollars\n"
example_1_string = "Example: message: ‘What is the dinner?’ intent_label: ‘Inquiry about In-flight Meals’\n"
example_2_string = "Example: message: ‘What city is the airport in?’ intent_label: ‘Airport Location Inquiry’\n"
closing_string = "Return the label that best matches the message\n"

# [Many more types of substrings, and variations of each would be listed here before we begin combining them into the final prompt strings]
if llm == 'gpt-4o-mini':
    system_prompt = role_string_1 + task_string + input_string + output_string
elif llm == 'mistral_tiny':
    pass
