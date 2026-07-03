simba_prog.save("prog_v1.json", save_program=False)
simba_prog2 = dspy.Predict(ClosedIntentSignature)
simba_prog2.load("prog_v1.json")
print(simba_prog2(message="What meals are served on the 3pm flight to Madrid?")

