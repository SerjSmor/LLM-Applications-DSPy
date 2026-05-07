import numpy as np
import dspy
from datasets import load_dataset

ATIS_INTENT_MAPPING = {
    'abbreviation': "Abbreviation and Fare Code Meaning Inquiry",
    'aircraft': "Aircraft Type Inquiry",
    # 'aircraft+flight+flight_no':   "",
    'airfare': "Airfare and Fees Questions",
    # 'airfare+flight_time':         "",
    'airline': "Airline Information Request",
    # 'airline+flight_no':           "",
    'airport': "Airport Information and Queries",
    'capacity': "Aircraft Seating Capacity Inquiry",
    'Cheapest': "Cheapest Fare Inquiry",
    'city': "Airport Location Inquiry",
    'distance': "Airport Distance Inquiry",
    'flight': "Flight Booking Request",
    # 'flight+airfare':              "",
    'flight_no': "Flight Number Inquiry",
    'flight_time': "Time Inquiry",
    'ground_fare': "Ground Transportation Cost Inquiry",
    'ground_service': "Ground Transportation Inquiry",
    # 'ground_service+ground_fare':  "Airport Ground Transportation and Cost Query",
    'meal': "Inquiry about In-flight Meals",
    'quantity': "Flight Quantity Inquiry",
    'restriction': "Flight Restriction Inquiry"
}


def create_examples_from_set(set_name, n=-1):
    ds = load_dataset("tuetschek/atis")
    ds.set_format(type='pandas')
    df = ds[set_name][:]
    df['intent'] = df['intent'].map(ATIS_INTENT_MAPPING)
    df = df.dropna(subset='intent')
    if n > 0:
        df = df.sample(n=n, random_state=42)

    examples = []
    for index in df.index:
        row = df.loc[index]
        examples.append(
            dspy.Example(message=row['text'],
                         intent_label=row['intent']).with_inputs('message')
        )
    return examples


train_val_set = create_examples_from_set('test')
np.random.shuffle(train_val_set)
train_set = train_val_set[:100]
val_set = train_val_set[100:400]

dev_test_set = create_examples_from_set('train')
np.random.shuffle(dev_test_set)
dev_set = dev_test_set[:1000]
test_set = dev_test_set[1000:2000]

if __name__ == '__main__':
    print(len(train_set), len(val_set), len(dev_set), len(test_set))
