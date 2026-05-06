import dspy
from dspy import Prediction

from common.consts import unique_intents


def validate_answer(example: dspy.Example, prediction: dspy.Prediction, trace=None):
    return example.intent_label.lower() == prediction.intent_label.strip().lower()


booking_flight_example = dspy.Example(
    message="I want to book a flight",
    labels=unique_intents,
    intent_label='Flight Booking Request').with_inputs("message", "labels")
booking_flight_prediction = Prediction(intent_label='Airport Distance Inquiry')

print(validate_answer(booking_flight_example, booking_flight_prediction))
