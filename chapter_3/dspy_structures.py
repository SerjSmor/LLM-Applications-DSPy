import dspy
from typing import List, Literal


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
    intent_label: Literal['Flight Booking Request',
                          'Airfare and Fees Questions',
                          'Ground Transportation Inquiry',
                          'Inquiry about In-flight Meals',
                          'Airport Information and Queries',
                          'Airline Information Request',
                          'Time Inquiry',
                          'Airport Location Inquiry',
                          'Ground Transportation Cost Inquiry',
                          'Flight Quantity Inquiry',
                          'Abbreviation and Fare Code Meaning Inquiry',
                          'Airport Distance Inquiry',
                          'Aircraft Type Inquiry',
                          'Aircraft Seating Capacity Inquiry',
                          'Flight Number Inquiry'
    ] = dspy.OutputField()