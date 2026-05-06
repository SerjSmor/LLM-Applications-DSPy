import dspy

from common.consts import unique_intents

turbo = dspy.LM(model='gpt-4o-mini')
dspy.settings.configure(lm=turbo)
generator = dspy.Predict("intent -> example_question")

synethetic_examples = []
for intent in unique_intents:
    for _ in range(10):
        prediction = generator(intent=intent)
        synethetic_examples.append(
            dspy.Example(message=prediction.example_question,
                         labels=unique_intents,
                         intent_label=intent
                         ).with_inputs('message', 'labels')
        )
