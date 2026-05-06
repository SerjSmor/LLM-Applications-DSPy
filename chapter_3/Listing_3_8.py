import dspy
from typing import List
from dotenv import load_dotenv

load_dotenv()

class DynamicIntentSignature(dspy.Signature):
    """
    Predict the label that best matches the message if any match well.
    If none match well, return None. #A
    """
    message: str = dspy.InputField()
    labels: List[str] = dspy.InputField()
    intent_label: str = dspy.OutputField()


labels = ['Food Inquiry', 'Airport Location Inquiry']  # B
lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)
classifier = dspy.Predict(DynamicIntentSignature)
prediction = classifier(message="I'd like to rebook a flight",
                        labels=labels)
print(prediction)
