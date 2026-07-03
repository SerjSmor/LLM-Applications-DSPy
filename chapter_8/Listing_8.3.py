
import dspy

def create_examples(df):
    examples = []
    for i in range(len(df)):
        example = dspy.Example({
            'dialog':df.iloc[i]['dialogue'],
            'summary':df.iloc[i]['summary']
        }).with_inputs('dialog')
        examples.append(example)
    return examples 

test_examples = create_examples(df_test)
train_examples = create_examples(df_train)

summarizer_train_data = train_examples[:50]
summarizer_val_data   = train_examples[50:150]
summarizer_test_data  = test_examples[:500]

print(len(summarizer_train_data), len(summarizer_val_data), len(summarizer_test_data))
