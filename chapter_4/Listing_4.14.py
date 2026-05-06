import time
from tqdm import tqdm
import numpy as np
from openai import OpenAI
import dspy

from chapter_4.Listing_4_5 import dev_set
from common.consts import unique_intents
from common.utils import validate_answer


def classify_message_prompt(message: str, system_prompt: str):
    model = 'gpt-4o-mini'
    client = OpenAI()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages
    )
    return completion.choices[0].message.content


def evaluate_openai_manual_prompt(system_prompt):
    scores = []
    start_time = time.time()
    for example in tqdm(dev_set):
        prediction = classify_message_prompt(example.message, system_prompt)
        score = validate_answer(example, dspy.Prediction(intent_label=prediction))
        scores.append(score)
    end_time = time.time()
    print(f"Accuracy: {np.mean(scores)}, total time: {end_time - start_time}")


system_prompt = f'''You are an expert of intent classification. Your task is to classify the intent of customer messages of an airline company into one of the provided labels. 
Input: customer message
Output: One of the following classes: {",".join(unique_intents)}'''

evaluate_openai_manual_prompt(system_prompt)
