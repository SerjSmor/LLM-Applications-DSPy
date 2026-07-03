import dspy
import copy
import numpy as np

def create_examples(df):
    examples = []
    for i in range(len(df)):
        example = dspy.Example({
            'dialog':df.iloc[i]['dialogue'],
            'summary':df.iloc[i]['summary'],
            'true_score': 1.0 
        }).with_inputs('dialog', 'summary') 
        examples.append(example)
    return examples 

test_examples = create_examples(df_test)
train_examples = create_examples(df_train)
np.random.shuffle(train_examples)
np.random.shuffle(test_examples)

judge_train_data = train_examples[150:200] 
judge_val_data   = train_examples[200:300]
judge_test_data  = test_examples[500:1000] 

neg_examples = copy.deepcopy(judge_val_data) 
for i in range(99):
    neg_examples[i]['summary'] = neg_examples[i+1]['summary']
    neg_examples[i]['true_score'] = 0.0
judge_val_data = judge_val_data + neg_examples

neg_examples = copy.deepcopy(judge_test_data) 
for i in range(499):
    neg_examples[i]['summary'] = neg_examples[i+1]['summary']
    neg_examples[i]['true_score'] = 0.0
judge_test_data = judge_test_data + neg_examples

print(len(judge_train_data), len(judge_val_data), len(judge_test_data))
