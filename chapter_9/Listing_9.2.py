from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-small-en-v1.5')

embedding = model.encode("Pricing Plans: Creating Discount Coupons")
print(type(embedding))
print(embedding.shape)
print(embedding)