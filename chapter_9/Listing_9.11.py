
import dspy
import pandas as pd
from sentence_transformers import SentenceTransformer

class SupportTurn(dspy.Signature): 
    """
    Answer a support request using retrieval when needed. Be careful that the 
    answer returned is correct based on the related articles retrieved.
    """
    user_request: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Helpful and accurate response to the user")

class SupportBackend: 
    def __init__(self, retriever: dspy.Module, kb_df: pd.DataFrame):
        self.retriever = retriever
        self.kb_df = kb_df

    def retrieve_article_titles_and_ids(self, title_query: str) -> str: 
        """Search KB articles by title query. Returns matching titles and IDs."""
        result = self.retriever(title_query)
        if not result.passages:
            return "No related KB articles found."
        lines = []
        for title, article_idx in zip(result.passages, result.indices):
            lines.append(f"- {title} (id={article_idx})")
        return "Matching articles:\n" + "\n".join(lines)

    def read_article(self, article_id: str) -> str: 
        """Read the full content of a KB article by its ID."""
        #article = self.kb_df[self.kb_df["id"] == article_id]
        article = self.kb_df.iloc[int(article_id)]
        if article.empty:
            return f"No article found with id '{article_id}'."
        return f"{article['title']}\n\n{article['contents']}"

lm = dspy.LM("openai/gpt-4.1-mini", cache=False)
dspy.configure(lm=lm)

model = SentenceTransformer('BAAI/bge-small-en-v1.5')
embedder = dspy.Embedder(model.encode) 
corpus = kb_df['title'].to_list()
retriever = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=5)

backend = SupportBackend(retriever=retriever, kb_df=kb_df) 

react_agent = dspy.ReAct( 
    signature=SupportTurn,
    tools=[backend.retrieve_article_titles_and_ids, backend.read_article],
    max_iters=4,
)

user_request = dataset_df.iloc[0]["question"]
result = react_agent(user_request=user_request)

print(f"Request: {user_request}")
print(f"Final answer: {result.answer}")
print("-" * 70)
print("Trajectory:")
print(getattr(result, "trajectory", "No trajectory found"))
