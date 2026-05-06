import dspy
import pandas as pd
from datasets import load_dataset

from common.consts import ATIS_INTENT_MAPPING

def create_examples_from_set(set_name, n=-1):
    ds = load_dataset("tuetschek/atis")
    ds.set_format(type='pandas')
    df: pd.DataFrame = ds[set_name][:]
    df['intent'] = df['intent'].map(ATIS_INTENT_MAPPING)
    df = df.dropna(subset='intent')
    unique_intents = df['intent'].unique()
    if n > 0:
        df = df.sample(n=n, random_state=42)

    examples = []
    for index in df.index:
        row = df.loc[index]
        examples.append(
            dspy.Example(message=row['text'],
                         labels=unique_intents,
                         intent_label=row['intent']).with_inputs('message', 'labels')
        )
    return examples


test_set = create_examples_from_set('test', 100)
