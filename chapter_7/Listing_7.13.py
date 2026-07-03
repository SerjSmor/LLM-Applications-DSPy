
from typing import List

class IntentClassifierSignature(dspy.Signature): 
   """
   Predict the label that best matches the message if any match well. 
   If none match well, return None.
   """
   message: str      = dspy.InputField()
   labels: List[str] = dspy.InputField()
   intent_label: str = dspy.OutputField()

class IntentClassifier(dspy.Module):
   def __init__(self, signature):
      super().__init__()
      self.toxic_classifier = dspy.Predict("query -> is_toxic: bool") 
      self.classifier = dspy.Predict(signature) 

   def forward(self, **kwargs):
      if self.toxic_classifier(query=kwargs['message']).is_toxic:
          print("Identified toxic comment")
          return None
      return self.classifier(**kwargs)

lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)
classifier = IntentClassifier(IntentClassifierSignature)
labels = ["Booking", "Rebooking", "Meals"] 
prediction = classifier(message="I'd like to rebook a flight",
                        labels=labels) 
print(prediction)
