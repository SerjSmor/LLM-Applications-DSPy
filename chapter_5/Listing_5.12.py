import dspy
from dspy import KNNFewShot
from sentence_transformers import SentenceTransformer

from chapter_5.Listing_5_1 import ClosedIntentSignature, lm
from chapter_5.Listing_5_2 import train_set, dev_set

encoder_func = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L12-V2",
    token=False).encode

knn_optimizer = KNNFewShot(
    k=3,
    trainset=train_set,
    vectorizer=dspy.Embedder(encoder_func)
)

optimized_module = knn_optimizer.compile(dspy.Predict(ClosedIntentSignature))
prediction = optimized_module(**dev_set[0].with_inputs())
print(lm.inspect_history(n=1))
