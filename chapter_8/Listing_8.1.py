
import dspy
validation_set = [
  dspy.Example({ 
   "document": "[the text of The Rime of the Ancient Mariner]", 
   "summary_1": "A sailor stops a wedding guest to recount his cursed journey after he  thoughtlessly shoots an albatross, a bird of good omen, bringing death to his shipmates and himself",
   "summary_2": "A sailor impulsively kills an albatross, leading to a supernatural curse that dooms his ship and crew, leaving him isolated and tormented until he learns to love and respect all of God's creatures."}).with_inputs("document"),

  dspy.Example({ 
   "document": "[the text of Howl]", 
   "summary_1": "Howl passionately decries the destructive forces of society (Moloch) on the best minds of my generation",
   "summary_2": "lamenting the destruction of the best minds of his generation by madness, conformity, and societal repression"}).with_inputs("document"),] 
