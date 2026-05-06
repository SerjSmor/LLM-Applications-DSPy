import dspy

example_1 = dspy.Example({
   "message": "I would like to find a flight from charlotte to las vegas that makes at "   
              "most one stop inbetween",
   "labels": unique_intents,
   "intent_label": "Flight Booking Request"}).with_inputs("message", "labels")
