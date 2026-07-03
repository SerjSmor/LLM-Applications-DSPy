
import dspy
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-small-en-v1.5')
embedder = dspy.Embedder(model.encode) 

question = "How can I add discounts to my service prices when customers pay for a plan?"

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
