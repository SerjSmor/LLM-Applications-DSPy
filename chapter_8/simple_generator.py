#!/home/serj/dev/prompt_programming_with_dspy/venv/bin/python
"""
Simple Wix Answer Generator CLI Script

A command-line interface for inference and evaluation using title, paragraph,
or HyDE retrievers. Results are logged to MLflow. Uses DSPy SemanticF1 as the
evaluation metric.

Usage:
    python chapter_8/simple_generator.py --mode inference --index 0
    python chapter_8/simple_generator.py --mode evaluate --num-examples 10
    python chapter_8/simple_generator.py --mode evaluate --retriever title
    python chapter_8/simple_generator.py --mode evaluate --retriever hyde
    python chapter_8/simple_generator.py --mode evaluate --retriever hyde --load-model chapter_8/optimized_mipro_hyde.pkl
"""

import argparse
from datetime import datetime

import datasets
import dspy
import mlflow
import pandas as pd
from sentence_transformers import SentenceTransformer

from retrieval import HyDERetriever


# =============================================================================
# Retrievers
# =============================================================================

class RetrievalByTitle(dspy.Module):
    """Retriever that searches KB articles by title similarity, returns full article content."""
    
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
        # Store article content for retrieval
        self.row_index_to_content = {i: f"{kb_df.iloc[i]['title']}\n{kb_df.iloc[i]['contents']}" 
                                      for i in range(len(kb_df))}
        
    def forward(self, question: str):
        prediction = self.search(question)
        indices = [int(index) for index in prediction.indices]
        prediction_articles_ids = [self.row_index_to_article_id[idx] for idx in indices]
        # Return actual article content, not just titles
        passages = [self.row_index_to_content[idx] for idx in indices]
        return dspy.Prediction(
            passages=passages,
            article_ids=prediction_articles_ids
        )


class RetrievalByParagraph(dspy.Module):
    """Retriever that searches KB articles by paragraph content similarity."""
    
    def __init__(self, chunks_df: pd.DataFrame, embedder_name: str = 'BAAI/bge-small-en-v1.5', 
                 k: int = 5):
        self.k = k
        self.chunks_df = chunks_df
        model = SentenceTransformer(embedder_name)
        embedder = dspy.Embedder(model.encode)
        
        corpus = chunks_df['content'].to_list()
        print(f"Paragraph retriever: {len(corpus)} chunks")
        self.search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=k)
        self.row_index_to_chunk = {i: chunks_df.iloc[i].to_dict() for i in range(len(chunks_df))}
        
    def forward(self, question: str):
        prediction = self.search(question)
        indices = [int(index) for index in prediction.indices]
        
        retrieved_chunks = [self.row_index_to_chunk[idx] for idx in indices]
        prediction_article_ids = list(dict.fromkeys([chunk['article_id'] for chunk in retrieved_chunks]))
        
        return dspy.Prediction(
            passages=prediction.passages,
            article_ids=prediction_article_ids,
            chunks=retrieved_chunks
        )


# =============================================================================
# Answer Generator
# =============================================================================

class AnswerGenerator(dspy.Signature):
    """Generate a helpful answer to the user's question based on retrieved KB articles."""
    question: str = dspy.InputField(desc="User's question about Wix")
    relevant_articles: list[str] = dspy.InputField(desc="Retrieved KB article passages")
    answer: str = dspy.OutputField(desc="Helpful answer based on the retrieved articles")


class SimpleWixAnswerGenerator(dspy.Module):
    """Simple end-to-end answer generator using title and paragraph retrievers."""
    
    def __init__(self, retriever: dspy.Module):
        """
        Args:
            retriever: Any retriever module that returns passages and article_ids.
                       Can be RetrievalByTitle, RetrievalByParagraph, or CombinedRetriever.
        """
        self.retriever = retriever
        self.answer_generator = dspy.ChainOfThought(AnswerGenerator)

    def forward(self, question: str):
        # Step 1: Retrieve relevant articles
        retrieval_result = self.retriever(question)
        
        # Step 2: Generate answer based on retrieved passages
        answer_result = self.answer_generator(
            question=question,
            relevant_articles=retrieval_result.passages
        )
        
        return dspy.Prediction(
            answer=answer_result.answer,
            reasoning=answer_result.reasoning,
            passages=retrieval_result.passages,
            article_ids=retrieval_result.article_ids
        )


# =============================================================================
# Metrics
# =============================================================================

def retrieval_success_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None):
    """Metric: checks if all expected articles were retrieved."""
    expected = set(example.article_ids)
    predicted = set(prediction.article_ids) if prediction.article_ids else set()
    return 1.0 if expected.issubset(predicted) else 0.0


def create_semantic_f1_metric():
    """Create the DSPy SemanticF1 metric wrapper."""
    from dspy.evaluate import SemanticF1
    
    semantic_f1 = SemanticF1()
    
    def metric(example: dspy.Example, prediction: dspy.Prediction, trace=None):
        """Wrapper metric that adapts our field names to SemanticF1's expected format."""
        
        # Create adapted example and prediction with 'response' field
        adapted_example = dspy.Example(
            question=example.question,
            response=example.answer  # SemanticF1 expects 'response'
        )
        adapted_pred = dspy.Prediction(
            response=prediction.answer  # SemanticF1 expects 'response'
        )
        
        try:
            score = semantic_f1(
                example=adapted_example,
                pred=adapted_pred,
                trace=trace
            )
            return score
        except Exception as e:
            print(f"SemanticF1 error: {e}")
            return 0.0
    
    return metric


# =============================================================================
# Data Loading
# =============================================================================

def load_data(max_chunks: int = None, max_articles: int = None):
    """Load WixQA dataset with optional size limits."""
    print("Loading WixQA dataset...")
    dataset = datasets.load_dataset('Wix/WixQA', 'wixqa_expertwritten')
    kb_dataset = datasets.load_dataset('Wix/WixQA', 'wix_kb_corpus')
    
    dataset.set_format(type='pandas')
    kb_dataset.set_format(type='pandas')
    
    train_df = dataset['train'][:]
    kb_df = kb_dataset['train'][:]
    
    # Limit KB articles if specified
    if max_articles and max_articles < len(kb_df):
        kb_df = kb_df.head(max_articles)
        print(f"Limited to {max_articles} KB articles")
    
    print(f"Train examples: {len(train_df)}")
    print(f"KB articles: {len(kb_df)}")
    
    # Create chunks from articles
    def chunk_article(article_id: str, title: str, content: str) -> list[dict]:
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        chunks = []
        for i, para in enumerate(paragraphs):
            chunks.append({
                'chunk_id': f"{article_id}_{i}",
                'article_id': article_id,
                'paragraph_idx': i,
                'content': f"{title}\n{para}",
                'title': title
            })
        return chunks
    
    all_chunks = []
    for _, row in kb_df.iterrows():
        chunks = chunk_article(row['id'], row['title'], row['contents'])
        all_chunks.extend(chunks)
        if max_chunks and len(all_chunks) >= max_chunks:
            all_chunks = all_chunks[:max_chunks]
            break
    
    chunks_df = pd.DataFrame(all_chunks)
    print(f"Total chunks: {len(chunks_df)}")
    
    return train_df, kb_df, chunks_df


def create_examples(train_df: pd.DataFrame) -> list[dspy.Example]:
    """Create DSPy examples from training dataframe."""
    examples = []
    for _, row in train_df.iterrows():
        if row['question']:
            examples.append(
                dspy.Example(
                    question=row['question'],
                    article_ids=row['article_ids'],
                    answer=row['answer']
                ).with_inputs('question')
            )
    return examples


def setup_retrievers(chunks_df: pd.DataFrame, kb_df: pd.DataFrame, 
                     title_k: int = 5, paragraph_k: int = 5):
    """Initialize the retrievers."""
    title_retriever = RetrievalByTitle(kb_df=kb_df, k=title_k)
    paragraph_retriever = RetrievalByParagraph(chunks_df=chunks_df, k=paragraph_k)
    
    return title_retriever, paragraph_retriever


# =============================================================================
# CLI Modes
# =============================================================================

def run_inference(generator: SimpleWixAnswerGenerator, examples: list[dspy.Example], 
                  kb_df: pd.DataFrame, index: int):
    """Run inference on a single example."""
    if index >= len(examples):
        print(f"Error: Index {index} out of range. Dataset has {len(examples)} examples.")
        return
    
    example = examples[index]
    print("=" * 70)
    print(f"INFERENCE - Example {index}")
    print("=" * 70)
    print(f"\nQuestion: {example.question}")
    print(f"\nExpected article IDs: {example.article_ids}")
    
    # Get expected titles
    expected_articles = kb_df[kb_df['id'].isin(example.article_ids)]
    if not expected_articles.empty:
        print("Expected articles:")
        for _, art in expected_articles.iterrows():
            print(f"  - {art['title']}")
    
    expected_answer = example.answer
    print(f"\nExpected answer: {expected_answer[:200]}..." if len(expected_answer) > 200 else f"\nExpected answer: {expected_answer}")
    
    print("\n" + "-" * 70)
    print("Running SimpleWixAnswerGenerator...")
    print("-" * 70)
    
    prediction = generator(example.question)
    
    print(f"\nRetrieved articles ({len(prediction.article_ids)}):")
    for i, passage in enumerate(prediction.passages):
        print(f"  {i+1}. {passage[:80]}...")
    
    print(f"\nGenerated Answer:")
    print("-" * 40)
    print(prediction.answer)
    print("-" * 40)
    
    print(f"\nReasoning:")
    reasoning = prediction.reasoning if hasattr(prediction, 'reasoning') and prediction.reasoning else "N/A"
    print(reasoning[:300] if len(reasoning) > 300 else reasoning)
    
    # Compute metrics
    retrieval_success = retrieval_success_metric(example, prediction)
    
    print("\n" + "=" * 70)
    print(f"RETRIEVAL SUCCESS: {'YES' if retrieval_success else 'NO'}")
    print(f"Predicted article IDs: {prediction.article_ids}")
    print("=" * 70)


def run_evaluate(generator: SimpleWixAnswerGenerator, examples: list[dspy.Example], 
                 kb_df: pd.DataFrame, num_examples: int,
                 retriever_type: str = "paragraph", k: int = 5, model: str = "openai/gpt-4.1-mini",
                 experiment_name: str = "simple_generator_evaluation", load_model: str = None):
    """Run evaluation on a subset of examples with MLflow logging. Uses SemanticF1 metric."""
    # Adjust experiment name for optimized models
    if load_model:
        experiment_name = f"{experiment_name}_optimized"
    
    print("=" * 70)
    print(f"EVALUATION - {num_examples} examples")
    print(f"Metric: SemanticF1")
    if load_model:
        print(f"Optimized model: {load_model}")
    print(f"MLflow Experiment: {experiment_name}")
    print("=" * 70)
    
    # Take the last N examples as the test set
    eval_examples = examples[-num_examples:]
    
    # Always use SemanticF1 metric
    print("Initializing DSPy SemanticF1 metric...")
    metric = create_semantic_f1_metric()
    
    # Log parameters
    mlflow.log_param("num_examples", num_examples)
    mlflow.log_param("metric", "semantic_f1")
    mlflow.log_param("retriever_type", retriever_type)
    mlflow.log_param("k", k)
    mlflow.log_param("model", model)
    mlflow.log_param("optimized", load_model is not None)
    if load_model:
        mlflow.log_param("model_path", load_model)
    
    # Run evaluation
    print(f"\nStarting evaluation at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = datetime.now()
    
    evaluator = dspy.Evaluate(
        devset=eval_examples,
        num_threads=4,
        display_progress=True,
        max_errors=10
    )
    
    result = evaluator(generator, metric=metric)
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Evaluation finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Evaluation duration: {duration}")
    
    # Log metrics
    mlflow.log_metric("semantic_f1_score", result.score)
    
    # Collect per-example results for detailed logging
    results_data = []
    for example, prediction, score in result.results:
        expected_article_ids = list(example.article_ids) if hasattr(example.article_ids, '__iter__') else [example.article_ids]
        predicted_article_ids = prediction.article_ids if prediction.article_ids else []
        
        # Get article titles
        expected_titles = []
        for art_id in expected_article_ids:
            art_row = kb_df[kb_df['id'] == art_id]
            if not art_row.empty:
                expected_titles.append(art_row.iloc[0]['title'])
        
        predicted_titles = []
        for art_id in predicted_article_ids:
            art_row = kb_df[kb_df['id'] == art_id]
            if not art_row.empty:
                predicted_titles.append(art_row.iloc[0]['title'])
        
        results_data.append({
            'question': example.question[:200],
            'expected_answer': example.answer[:200] if example.answer else '',
            'predicted_answer': prediction.answer[:200] if prediction.answer else '',
            'score': score,
            'expected_articles': '; '.join(expected_titles),
            'predicted_articles': '; '.join(predicted_titles),
            'retrieval_success': '✓' if set(expected_article_ids).issubset(set(predicted_article_ids)) else '✗'
        })
    
    # Log results table
    results_df = pd.DataFrame(results_data)
    mlflow.log_table(data=results_df, artifact_file="evaluation_results.json")
    
    # Log summary statistics
    retrieval_successes = sum(1 for r in results_data if r['retrieval_success'] == '✓')
    mlflow.log_metric("retrieval_success_rate", retrieval_successes / len(results_data) * 100)
    
    print("\n" + "=" * 70)
    print(f"SEMANTIC F1 SCORE: {result.score:.2f}%")
    print(f"RETRIEVAL SUCCESS RATE: {retrieval_successes}/{len(results_data)} ({retrieval_successes/len(results_data)*100:.1f}%)")
    print(f"Results logged to MLflow experiment: {experiment_name}")
    print("=" * 70)
    
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Simple Wix Answer Generator CLI")
    parser.add_argument("--mode", choices=["inference", "evaluate"], 
                        required=True, help="Mode to run")
    parser.add_argument("--index", type=int, default=0, 
                        help="Example index for inference mode")
    parser.add_argument("--num-examples", type=int, default=50, 
                        help="Number of examples for evaluation")
    parser.add_argument("--max-chunks", type=int, default=None, 
                        help="Max corpus chunks to load")
    parser.add_argument("--max-articles", type=int, default=None, 
                        help="Max KB articles to load")
    parser.add_argument("--model", type=str, default="openai/gpt-4.1-mini", 
                        help="LLM model to use")
    parser.add_argument("--retriever", choices=["title", "paragraph", "hyde"], 
                        default="paragraph", help="Which retriever to use")
    parser.add_argument("--k", type=int, default=5, 
                        help="Number of passages to retrieve")
    parser.add_argument("--experiment-name", type=str, default="simple_generator_evaluation",
                        help="MLflow experiment name")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to load optimized HyDE model (e.g., chapter_8/optimized_mipro_hyde.pkl)")
    
    args = parser.parse_args()
    
    # Configure LLM
    print(f"Configuring LLM: {args.model}")
    lm = dspy.LM(args.model, cache=False)
    dspy.configure(lm=lm)
    
    # Determine if we should start an MLflow run early to capture setup duration
    if args.mode == "evaluate":
        # Prepare experiment name (consistent with run_evaluate)
        experiment_name = f"{args.experiment_name}_{args.retriever}"
        if args.load_model:
            experiment_name = f"{experiment_name}_optimized"
            
        mlflow.set_experiment(experiment_name)
        run_ctx = mlflow.start_run(run_name=f"{args.retriever}_semantic_f1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    else:
        from contextlib import nullcontext
        run_ctx = nullcontext()

    with run_ctx:
        # Load data
        train_df, kb_df, chunks_df = load_data(
            max_chunks=args.max_chunks, 
            max_articles=args.max_articles
        )
        
        # Create examples
        examples = create_examples(train_df)
        print(f"Created {len(examples)} examples")
        
        # Setup retrievers
        title_retriever, paragraph_retriever = setup_retrievers(
            chunks_df, kb_df, title_k=args.k, paragraph_k=args.k
        )
        
        # Select retriever based on argument
        if args.retriever == "title":
            retriever = title_retriever
            print("Using: Title Retriever")
        elif args.retriever == "paragraph":
            retriever = paragraph_retriever
            print("Using: Paragraph Retriever")
        else:  # hyde
            print("Using: HyDE Retriever")
            retriever = HyDERetriever(
                chunks_df=chunks_df,
                kb_df=kb_df,
                num_titles=20,
                num_hypotheses=3,
                k_per_hypothesis=5,
                final_k=args.k
            )
            
            # Load optimized model if specified
            if args.load_model:
                print(f"Loading optimized model from: {args.load_model}")
                retriever.load(args.load_model)
                print("Optimized model loaded successfully!")
        
        # Create generator
        generator = SimpleWixAnswerGenerator(retriever=retriever)
        
        # Run mode
        if args.mode == "inference":
            run_inference(generator, examples, kb_df, args.index)
        elif args.mode == "evaluate":
            # Include retriever type in experiment name
            experiment_name = f"{args.experiment_name}_{args.retriever}"
            run_evaluate(
                generator=generator,
                examples=examples,
                kb_df=kb_df,
                num_examples=args.num_examples,
                retriever_type=args.retriever,
                k=args.k,
                model=args.model,
                experiment_name=experiment_name,
                load_model=args.load_model
            )


if __name__ == "__main__":
    main()
