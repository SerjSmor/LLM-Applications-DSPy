import dspy
import pandas as pd
from chapter_9.listing_9_4 import load_data, RetrievalByTitle


class AnswerGenerator(dspy.Signature):
    """Generate a helpful answer to the user's question based on the retrieved KB articles."""
    question: str = dspy.InputField(desc="User's question about Wix")
    relevant_articles: list[str] = dspy.InputField(desc="Retrieved KB article contents")
    answer: str = dspy.OutputField(desc="Helpful answer based on the retrieved articles")


class RAG(dspy.Module):
    """End-to-end answer generator combining retrieval and generation."""
    def __init__(self, retriever: dspy.Module, kb_df: pd.DataFrame):
        self.retriever = retriever
        self.kb_df = kb_df
        self.generate_answer = dspy.ChainOfThought(AnswerGenerator)

    def forward(self, question: str):
        # Step 1: Retrieve relevant article titles and IDs
        retrieval_result = self.retriever(question)

        # Step 2: Look up full article content using retrieved article_ids
        articles = []
        for article_id in retrieval_result.article_ids:
            article = self.kb_df[self.kb_df['id'] == article_id]
            if not article.empty:
                row = article.iloc[0]
                articles.append(f"{row['title']}\n{row['contents']}")

        # Step 3: Generate answer based on retrieved articles
        answer_result = self.generate_answer(
            question=question,
            relevant_articles=articles
        )

        return dspy.Prediction(
            answer=answer_result.answer,
            reasoning=answer_result.reasoning,
            passages=retrieval_result.passages,
            article_ids=retrieval_result.article_ids
        )

if __name__ == "__main__":
    # Configure LLM
    lm = dspy.LM("openai/gpt-4.1-mini", cache=False)
    dspy.configure(lm=lm)

    # Load data
    train_df, kb_df = load_data()

    # Build retriever and generator
    retriever = RetrievalByTitle(kb_df=kb_df)
    generator = RAG(retriever=retriever, kb_df=kb_df)

    # Run a single question
    question = "How can I add discounts to my service prices when customers pay for a plan?"
    prediction = generator(question)

    print("=" * 70)
    print(f"Question: {question}")
    print("=" * 70)
    print(f"\nRetrieved Articles ({len(prediction.article_ids)}):")
    for i, title in enumerate(prediction.passages):
        print(f"  {i+1}. {title}")

    print(f"\nGenerated Answer:")
    print("-" * 70)
    print(prediction.answer)
    print("-" * 70)
