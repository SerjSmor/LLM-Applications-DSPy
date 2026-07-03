
import dspy
import pandas as pd
from sentence_transformers import SentenceTransformer

class AnswerGenerator(dspy.Signature): 
    """
    Generate a helpful answer to the user's question based on the retrieved KB articles.
    Indicate if the question cannot be answered based on the retrieved articles. 
    """
    question: str = dspy.InputField(desc="User's question about Wix")
    relevant_articles: list[str] = dspy.InputField(desc="Retrieved KB article contents")
    answer: str = dspy.OutputField(desc="Helpful answer based on the retrieved articles")

class MultiStepRAG(dspy.Module): 
    """End-to-end answer generator combining retrieval and generation."""

    def __init__(self, retriever: dspy.Module, kb_df: pd.DataFrame, n_hops):
        self.retriever = retriever
        self.kb_df = kb_df
        self.max_hops = n_hops
        self.generate_search_term = dspy.ChainOfThought(
           "question -> vector_database_search_term") 
        self.generate_answer = dspy.ChainOfThought(AnswerGenerator) 
        self.evaluate_answer = dspy.ChainOfThought("question, answer -> good_answer:bool") 

    def forward(self, question: str):
        n_hops = 0        
        while n_hops < self.max_hops:
            search_term = self.generate_search_term(question=question)         
            print(f"{search_term=}")
            retrieval_result = self.retriever(search_term.vector_database_search_term) 

            articles = []
            for article_id in retrieval_result.indices: 
                title = self.kb_df.iloc[article_id]['title']
                article = self.kb_df.iloc[article_id]['contents']
                articles.append(title + "\n" + article)

            answer = self.generate_answer(question=question, relevant_articles=articles)
            print(f"{answer=}")
            good_answer = self.evaluate_answer(question=question, answer=answer)
            print(f"{good_answer=}")
            if good_answer.good_answer:
                return answer
            n_hops += 1
        return answer #

lm = dspy.LM("openai/gpt-4.1-mini", cache=False)
dspy.configure(lm=lm)

model = SentenceTransformer('BAAI/bge-small-en-v1.5')
embedder = dspy.Embedder(model.encode)
corpus = kb_df['title'].to_list()
retriever = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=5)

rag_prog = MultiStepRAG(retriever=retriever, kb_df=kb_df, n_hops=10)

question = "How can I add discounts to my service prices when customers pay for a plan?"
prediction = rag_prog(question)
print(prediction)
