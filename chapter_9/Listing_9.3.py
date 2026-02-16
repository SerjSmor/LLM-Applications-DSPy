import dspy
from sentence_transformers import SentenceTransformer
question = "How can I add discounts to my service prices when customers pay for a plan?"
embedder_name = 'BAAI/bge-small-en-v1.5'
model = SentenceTransformer(embedder_name)
embedder = dspy.Embedder(model.encode)

corpus = [
    "Pricing Plans: Creating Discount Coupons", 
    "Pricing Plans: Using Discount Coupons", 
    "Wix Events: About the Event Details and Registration Form Pages", 
    "Wix Restaurants Request: Changing the Alignment of Your Menus",
    "Adding Pinterest Content to Your Site"
]

search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=2)
prediction = search(question)
print(prediction)