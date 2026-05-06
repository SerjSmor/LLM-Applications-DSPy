import dspy

from chapter_4.Listing_4_5 import test_set
from common.consts import unique_intents

mini = dspy.LM(model='gpt-4o-mini')
dspy.settings.configure(lm=mini)
generator = dspy.Predict("user_message -> alternative_wording")

synthetic_examples = []
for original_example in test_set:
    for _ in range(2):
        prediction = generator(user_message=original_example.message)
        synthetic_examples.append(
            dspy.Example(
                message=prediction.alternative_wording,
                labels=unique_intents,
                intent_label=original_example.intent_label
            ).with_inputs('message', 'labels')
        )
