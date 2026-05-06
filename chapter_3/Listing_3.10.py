from typing import List
import dspy

class MultiIntentSignature(dspy.Signature):
   """
   Classify the message into one or more of the possible intent labels.
   """
   message: str = dspy.InputField()
   labels: List[str] = dspy.InputField()
   intent_label: str = dspy.OutputField(desc="Return the closest matches if any are "
      "reasonably close. Return as many as are close. Otherwise return None")

message = "I'd like to book a flight to Madrid from Berlin and order the pasta dinner."
labels = list(unique_intents)
lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)
classifier = dspy.Predict(MultiIntentSignature)
prediction = classifier(message=message, labels=labels)
print(prediction)


