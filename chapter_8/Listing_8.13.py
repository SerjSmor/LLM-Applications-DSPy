
import dspy
from dspy.evaluate import SemanticF1

lm = dspy.LM('openai/gpt-4.1-nano', cache=False) 
dspy.settings.configure(lm=lm)

test_example = dspy.Example( 
   question="What is the Rime of the Ancient Mariner about?", 
   response="A sailor stops a wedding guest to recount his cursed journey after he "
            "thoughtlessly shoots an albatross, a bird of good omen, bringing death to " 
            "shipmates and himself.") 

test_pred = dspy.Prediction(response=
   "A mariner recalls his voyage at sea, which was doomed after he killed a bird") #B

semantic_f1 = SemanticF1() 
semantic_f1(test_example, test_pred) 
