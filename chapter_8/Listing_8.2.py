
from datasets import load_dataset
import pandas as pd

ds = load_dataset("knkarthick/dialogsum")
ds.set_format(type='pandas')
df_train = ds['train'][:]
df_test = ds['test'][:]
