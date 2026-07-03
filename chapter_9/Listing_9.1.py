
import datasets

dataset = datasets.load_dataset('Wix/WixQA', 'wixqa_expertwritten')
dataset.set_format(type='pandas')
dataset_df = dataset['train'][:]
print(dataset_df.head(3))

kb_dataset = datasets.load_dataset('Wix/WixQA', 'wix_kb_corpus')
kb_dataset.set_format(type='pandas')
kb_df = kb_dataset['train'][:]
print(kb_df.head(2))
