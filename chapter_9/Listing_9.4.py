
import dspy
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-small-en-v1.5')
embedder = dspy.Embedder(model.encode)

question = "How can I add discounts to my service prices when customers pay for a plan?"

corpus = kb_df['title'].to_list() 
retriever = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=5) 
best_matches = retriever(question) 
indices = [int(index) for index in best_matches.indices] 
