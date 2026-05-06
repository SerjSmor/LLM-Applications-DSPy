import numpy as np

from common.utils import create_examples_from_set

train_val_set = create_examples_from_set('train', n=100)
np.random.shuffle(train_val_set)
train_set = train_val_set[:20]
val_set = train_val_set[20:]

dev_test_set = create_examples_from_set('test', n=100)
np.random.shuffle(dev_test_set)
dev_set = dev_test_set[:50]
test_set = dev_test_set[50:]
