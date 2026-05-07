import dspy
from dotenv import load_dotenv
from typing import List, Literal

load_dotenv()

lm = dspy.LM(model='gpt-4o-mini')
dspy.settings.configure(lm=lm)


def validate_answer(example: dspy.Example, prediction: dspy.Prediction, trace=None):
    return example.intent_label == prediction.intent_label


class IntentSignature(dspy.Signature):
    """
    Classify the message into one of the possible labels.
    """
    message: str = dspy.InputField()
    labels: List[str] = dspy.InputField()
    intent_label: str = dspy.OutputField()


class ClosedIntentSignature(dspy.Signature):
    """
    Classify the message into one of the possible labels.
    """
    message: str = dspy.InputField()
    intent_label: Literal[
        "Abbreviation and Fare Code Meaning Inquiry",
        "Aircraft Type Inquiry",
        "Airfare and Fees Questions",
        "Airline Information Request",
        "Airport Information and Queries",
        "Aircraft Seating Capacity Inquiry",
        "Cheapest Fare Inquiry",
        "Airport Location Inquiry",
        "Airport Distance Inquiry",
        "Flight Booking Request",
        "Flight Number Inquiry",
        "Time Inquiry",
        "Ground Transportation Cost Inquiry",
        "Ground Transportation Inquiry",
        "Inquiry about In-flight Meals",
        "Flight Quantity Inquiry",
        "Flight Restriction Inquiry"
    ] = dspy.OutputField()


intent_classifier = dspy.Predict(ClosedIntentSignature)
if __name__ == '__main__':
    prediction = intent_classifier(
        message="What meals are served on the 3pm flight to Madrid?")
