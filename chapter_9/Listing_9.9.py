
class ExampleDocumentGenerator(dspy.Signature): 
    """
    Generate an example of title of an article that could be found in a knowledge base
    that would provide an answer to the user's question.
    """
    question: str = dspy.InputField(desc="User's question about Wix")
    hypothetical_article_title: str = dspy.OutputField(
       desc="The title of a relevant article")

class HYDERAG(dspy.Module): 
    """End-to-end answer generator combining retrieval and generation."""

    def __init__(self, retriever: dspy.Module, kb_df: pd.DataFrame, n_hops):
        self.retriever = retriever
        self.kb_df = kb_df
        self.max_hops = n_hops
        self.hypothetical_article_title = dspy.ChainOfThought(ExampleDocumentGenerator)
        self.generate_answer = dspy.ChainOfThought(AnswerGenerator)

    def forward(self, question: str):
          articles = []
          article_indexes = []
          for n_hops in range(self.max_hops): 
              suggestion = self.hypothetical_article_title(question=question)          
              retrieval_result = self.retriever(suggestion.hypothetical_article_title) 

              for article_id in retrieval_result.indices: 
                  if article_id in article_indexes: 
                      continue
                  article_indexes.append(article_id)
                  title = self.kb_df.iloc[article_id]['title']
                  article = self.kb_df.iloc[article_id]['contents']
                  articles.append(title + "\n" + article)

          return self.generate_answer(question=question, relevant_articles=articles) 

rag_prog = HYDERAG(retriever=retriever, kb_df=kb_df, n_hops=10)
question = "How can I add discounts to my service prices when customers pay for a plan?"
prediction = rag_prog(question)
print(prediction)
