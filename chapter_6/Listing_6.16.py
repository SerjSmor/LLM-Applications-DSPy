
simba_prog.save("prog_v1", save_program=True)
simba_prog2 = dspy.load("prog_v1", allow_pickle=True)
print(simba_prog2(message="What meals are served on the 3pm flight to Madrid?"))
