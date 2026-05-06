import dspy
from typing import List
from dotenv import load_dotenv

load_dotenv()

class IntentSignature(dspy.Signature):
    """
    Classify the message into one of the possible labels.
    """
    message: str = dspy.InputField()
    labels: List[str] = dspy.InputField()
    intent_label: str = dspy.OutputField()


message = "I'd like to book a flight to Madrid from Berlin."
labels = list(unique_intents)
lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)
classifier = dspy.Predict(IntentSignature)
prediction = classifier(message=message, labels=labels)
print(prediction)
