import os
import time
from time import sleep
import dspy
from tqdm import tqdm

from chapter_3.dspy_structures import IntentSignature
from common.utils import create_examples_from_set, validate_answer

os.environ['MISTRAL_API_KEY'] = ""

lm = dspy.LM("mistral/mistral-tiny")
dspy.settings.configure(lm=lm)

# Initialize classifier and examples
classifier = dspy.ChainOfThought(IntentSignature)
examples = create_examples_from_set('test', n=100)
scores = []

# Running predictions and saving scores
start_time = time.time()
for example in tqdm(examples):
    sleep(1)
    prediction = classifier(**example.inputs())
    score = validate_answer(example, prediction)
    scores.append(score)

# Calculate average score
end_time = time.time()
average_score = sum(scores) / len(scores)
print(f"average score: {average_score}, total time: {end_time - start_time}")
