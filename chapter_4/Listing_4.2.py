import dspy

test_examples = [
    dspy.Example({
        "message": "I would like to find a flight from charlotte to las vegas that makes at "
                   "most one stop inbetween",
        "labels": unique_intents,
        "intent_label": "Flight Booking Request"}).with_inputs("message", "labels"),

    dspy.Example({
        "message": "What time does my flight to charlotte arrive?",
        "labels": unique_intents,
        "intent_label": "Time Inquiry"}).with_inputs("message", "labels")
]
