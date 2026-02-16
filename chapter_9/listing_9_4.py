import dspy
import pandas as pd
import datasets
from sentence_transformers import SentenceTransformer


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load WixQA dataset."""
    print("Loading WixQA dataset...")
    dataset = datasets.load_dataset('Wix/WixQA', 'wixqa_expertwritten')
    kb_dataset = datasets.load_dataset('Wix/WixQA', 'wix_kb_corpus')

    dataset.set_format(type='pandas')
    kb_dataset.set_format(type='pandas')

    train_df = dataset['train'][:]
    kb_df = kb_dataset['train'][:]

    return train_df, kb_df


class RetrievalByTitle(dspy.Module):
    """Retriever that searches KB articles by title similarity."""

    def __init__(self, kb_df: pd.DataFrame, embedder_name: str = 'BAAI/bge-small-en-v1.5',
                 k: int = 5):
        self.k = k
        self.kb_df = kb_df
        model = SentenceTransformer(embedder_name)
        embedder = dspy.Embedder(model.encode)

        corpus = kb_df['title'].to_list()
        print(f"Title retriever: {len(corpus)} articles")
        self.search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=k)
        self.row_index_to_article_id = {i: kb_df.iloc[i]['id'] for i in range(len(kb_df))}

    def forward(self, question: str):
        prediction = self.search(question)
        indices = [int(index) for index in prediction.indices]
        prediction_articles_ids = [self.row_index_to_article_id[idx] for idx in indices]
        return dspy.Prediction(
            passages=prediction.passages,
            article_ids=prediction_articles_ids
        )

if __name__ == "__main__":
    train_df, kb_df = load_data()
    retriever = RetrievalByTitle(kb_df=kb_df)
    question = "How can I add discounts to my service prices when customers pay for a plan?"
    prediction = retriever(question)
    print(prediction)
