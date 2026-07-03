
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

encoder_func = SentenceTransformer("sentence-transformers/all-MiniLM-L12-V2", 
                                   token=False).encode

v1 = encoder_func("the quick brown fox jumped over the lazy dog")
v2 = encoder_func("the lazy dog was jumped over by the fast brownish fox")
v3 = encoder_func("the really quick brown fox hopped over the dog")
v4 = encoder_func("a man is about to drown in the ocean of despair")

print(cosine_similarity([v1], [v1]))
print(cosine_similarity([v1], [v2]))
print(cosine_similarity([v1], [v3]))
print(cosine_similarity([v1], [v4]))
