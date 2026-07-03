
examples = [] 
for index in range(len(dataset_df)):
    row = dataset_df.iloc[index]
    examples.append(dspy.Example({'question': row['question'], 'answer':row['answer']}).with_inputs('question'))
print(f"Created {len(examples)} examples")

train_examples = examples[:50] 
test_examples = examples[50:] 
print(f"train set: {len(train_examples)} examples")
print(f"Test set: {len(test_examples)} examples")

corpus = kb_df['title'].to_list()
title_retriever = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=5)  

judge_model_name = "openai/gpt-4.1-mini" 
judge_lm = dspy.LM(judge_model_name, cache=False)

gen_lm = dspy.LM("openai/gpt-4.1-nano", cache=False) 
dspy.configure(lm=gen_lm)

rag_prog = RAG(retriever=title_retriever, kb_df=kb_df)
rag_prog.lm = gen_lm

llm_judge = LLMJudge()
llm_judge.lm = judge_lm 

evaluator = dspy.Evaluate( 
    devset=test_examples,
    num_threads=4,
    display_progress=True,
    max_errors=10
)

result = evaluator(rag_prog, metric=llm_judge) 
print(f"SCORE: {result.score:.2f}%")
