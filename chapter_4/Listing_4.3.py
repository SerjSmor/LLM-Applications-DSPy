import dspy

test_set_simple = [
    ("I would like to find a flight from charlotte to las vegas that makes at most stop inbetween",
     "Flight Booking Request"),

    ("What time does my flight to charlotte arrive",
     "Time Inquiry")
]

test_examples = [
    dspy.Example({
        "message": x[0],
        "labels": unique_intents,
        "intent_label": x[1]}).with_inputs("message", "labels")
    for x in test_set_simple
]
