
import dspy
import pandas as pd
from sentence_transformers import SentenceTransformer

class AnswerGenerator(dspy.Signature): 
    """
    Generate a helpful answer to the user's question based on the retrieved KB articles.
    """
    question: str = dspy.InputField(desc="User's question about Wix")
    relevant_articles: list[str] = dspy.InputField(desc="Retrieved KB article contents")
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField(desc="Helpful answer based on the retrieved articles")

class RAG(dspy.Module): 
    """End-to-end answer generator combining retrieval and generation."""

    def __init__(self, retriever: dspy.Module, kb_df: pd.DataFrame):
        self.session_history = dspy.History(messages=[])
        self.retriever = retriever
        self.kb_df = kb_df
        self.generate_answer = dspy.ChainOfThought(AnswerGenerator)

    def forward(self, question: str):
        retrieval_result = self.retriever(question) 

        articles = []
        for article_id in retrieval_result.indices: 
            title = self.kb_df.iloc[article_id]['title']
            article = self.kb_df.iloc[article_id]['contents']
            articles.append(title + "\n" + article)

        response = self.generate_answer(question=question, relevant_articles=articles, 
                                        history=self.session_history)
        self.session_history.messages.append({"question": question, **response})
        return response

lm = dspy.LM("openai/gpt-4.1-mini", cache=False)
dspy.configure(lm=lm)

model = SentenceTransformer('BAAI/bge-small-en-v1.5')
embedder = dspy.Embedder(model.encode)
corpus = kb_df['title'].to_list()
retriever = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=5)

rag_prog = RAG(retriever=retriever, kb_df=kb_df)

question = "How can I add discounts to my service prices when customers pay for a plan?"
prediction = rag_prog(question)
print(prediction)

question = "Okay, then specifically how would I do that?"
prediction = rag_prog(question)
print(prediction)
