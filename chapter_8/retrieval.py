#!/home/serj/dev/prompt_programming_with_dspy/venv/bin/python
"""
Retrieval Evaluation Script

Evaluates title, paragraph, or HyDE retriever using retrieval success metric.
Results are logged to MLflow.

Usage:
    python chapter_8/retrieval.py --retriever title
    python chapter_8/retrieval.py --retriever paragraph
    python chapter_8/retrieval.py --retriever hyde
    python chapter_8/retrieval.py --retriever hyde --load-model chapter_8/optimized_mipro_hyde.pkl
    python chapter_8/retrieval.py --retriever title --all    # Evaluate on all examples
    
    # Evaluate retrieval from a saved generator model (joint optimization output)
    python chapter_8/retrieval.py --retriever generator --load-model chapter_8/optimized_joint_generator.pkl
"""

import argparse
from datetime import datetime

import datasets
import dspy
import mlflow
import pandas as pd
from sentence_transformers import SentenceTransformer

from embeddings_with_scores import EmbeddingsWithScores


# =============================================================================
# Retrievers
# =============================================================================

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
# HyDE Retriever
# =============================================================================

class GenerateHypotheticalAnswers(dspy.Signature):
    """Generate hypothetical KB article paragraphs that would answer the user's question.
    Use the provided similar KB titles as guidance for correct terminology and topics.
    Each hypothesis should approach the question from a different angle."""
    
    question: str = dspy.InputField(desc="User's question about Wix")
    similar_titles: list[str] = dspy.InputField(desc="Top similar KB article titles for terminology guidance")
    num_hypotheses: int = dspy.InputField(desc="Number of hypotheses to generate")
    hypothetical_answers: list[str] = dspy.OutputField(
        desc="List of diverse hypothetical KB article paragraphs that answer this question"
    )


class HyDERetriever(dspy.Module):
    """Multi-Hypothesis HyDE retriever with score-based reranking and title context.
    
    Uses EmbeddingsWithScores for retrieval with similarity scores.
    
    1. Retrieves top-N similar titles for terminology guidance
    2. Generates N hypothetical answers per question using title context
    3. Searches each hypothesis, retrieving K results with scores
    4. Pools all results and reranks by score
    5. Returns top-K unique results
    """
    
    def __init__(self, chunks_df: pd.DataFrame, kb_df: pd.DataFrame,
                 embedder_name: str = 'BAAI/bge-small-en-v1.5',
                 num_titles: int = 20, num_hypotheses: int = 3, 
                 k_per_hypothesis: int = 5, final_k: int = 5):
        self.chunks_df = chunks_df
        self.kb_df = kb_df
        self.num_titles = num_titles
        self.num_hypotheses = num_hypotheses
        self.k_per_hypothesis = k_per_hypothesis
        self.final_k = final_k
        
        # Initialize embedder
        model = SentenceTransformer(embedder_name)
        embedder = dspy.Embedder(model.encode)
        
        # Title retriever (for terminology guidance)
        title_corpus = kb_df['title'].to_list()
        print(f"HyDE title retriever: {len(title_corpus)} titles")
        self.title_retriever = EmbeddingsWithScores(
            embedder=embedder, corpus=title_corpus, k=num_titles
        )
        
        # Paragraph retriever (main retrieval)
        chunk_corpus = chunks_df['content'].to_list()
        print(f"HyDE paragraph retriever: {len(chunk_corpus)} chunks")
        self.retriever = EmbeddingsWithScores(
            embedder=embedder, corpus=chunk_corpus, k=k_per_hypothesis
        )
        
        # Chunk mapping
        self.row_index_to_chunk = {i: chunks_df.iloc[i].to_dict() for i in range(len(chunks_df))}
        
        # HyDE generator (this is what gets optimized/loaded from pickle)
        self.generate_hypotheses = dspy.ChainOfThought(GenerateHypotheticalAnswers)
    
    def forward(self, question: str):
        # Step 0: Get similar titles for context
        title_prediction = self.title_retriever(question)
        similar_titles = title_prediction.passages[:self.num_titles]
        
        # Step 1: Generate multiple hypothetical answers with title context
        hypothesis_result = self.generate_hypotheses(
            question=question,
            similar_titles=similar_titles,
            num_hypotheses=self.num_hypotheses
        )
        hypotheses = hypothesis_result.hypothetical_answers
        
        # Ensure we have a list
        if isinstance(hypotheses, str):
            hypotheses = [hypotheses]
        
        # Step 2: Search each hypothesis and collect results with scores
        all_results = []
        for hyp in hypotheses:
            prediction = self.retriever(hyp)
            for passage, index, score in zip(
                prediction.passages, prediction.indices, prediction.scores
            ):
                all_results.append({
                    'index': index,
                    'score': score,
                    'passage': passage,
                    'hypothesis': hyp
                })
        
        # Step 3: Deduplicate by index, keeping highest score
        seen_indices = {}
        for r in all_results:
            idx = r['index']
            if idx not in seen_indices or r['score'] > seen_indices[idx]['score']:
                seen_indices[idx] = r
        
        # Step 4: Sort by score and take top-K
        unique_results = list(seen_indices.values())
        unique_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = unique_results[:self.final_k]
        
        # Step 5: Build output
        retrieved_chunks = [self.row_index_to_chunk[r['index']] for r in top_results]
        prediction_article_ids = list(dict.fromkeys([chunk['article_id'] for chunk in retrieved_chunks]))
        
        return dspy.Prediction(
            passages=[r['passage'] for r in top_results],
            article_ids=prediction_article_ids,
            chunks=retrieved_chunks,
            hypotheses=hypotheses,
            scores=[r['score'] for r in top_results]
        )


# =============================================================================
# Generator Retrieval Wrapper (for evaluating retrieval from saved generators)
# =============================================================================

class GeneratorRetrievalWrapper(dspy.Module):
    """Wrapper that extracts retrieval results from a full generator module.
    
    Used to evaluate retrieval success of saved generator models (like optimized_joint_generator.pkl).
    Calls the full generator but only returns retrieval-related fields for evaluation.
    """
    
    def __init__(self, generator):
        """
        Args:
            generator: A loaded generator module (e.g., SimpleWixAnswerGenerator)
        """
        self.generator = generator
    
    def forward(self, question: str):
        """Run the generator and return only retrieval results."""
        result = self.generator(question)
        return dspy.Prediction(
            passages=result.passages,
            article_ids=result.article_ids
        )


# =============================================================================
# Metrics
# =============================================================================

def retrieval_success_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None):
    """Metric: checks if all expected articles were retrieved."""
    expected = set(example.article_ids)
    predicted = set(prediction.article_ids) if prediction.article_ids else set()
    return 1.0 if expected.issubset(predicted) else 0.0


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


# =============================================================================
# Evaluation
# =============================================================================

def run_evaluate(retriever: dspy.Module, examples: list[dspy.Example], 
                 kb_df: pd.DataFrame, num_examples: int,
                 retriever_type: str, k: int,
                 experiment_name: str = "retrieval_evaluation",
                 load_model: str = None):
    """Run retrieval evaluation with MLflow logging."""
    
    # Include retriever type in experiment name (and optimized suffix if applicable)
    if load_model:
        full_experiment_name = f"{experiment_name}_{retriever_type}_optimized"
    else:
        full_experiment_name = f"{experiment_name}_{retriever_type}"
    
    print("=" * 70)
    print(f"RETRIEVAL EVALUATION - {num_examples} examples")
    print(f"Retriever: {retriever_type}")
    if load_model:
        print(f"Optimized model: {load_model}")
    print(f"k: {k}")
    print(f"MLflow Experiment: {full_experiment_name}")
    print("=" * 70)
    
    # Take the last N examples as the test set
    eval_examples = examples[-num_examples:]
    
    # Log parameters
    mlflow.log_param("num_examples", num_examples)
    mlflow.log_param("retriever_type", retriever_type)
    mlflow.log_param("k", k)
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
    
    result = evaluator(retriever, metric=retrieval_success_metric)
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Evaluation finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Evaluation duration: {duration}")
    
    # Log metrics
    mlflow.log_metric("retrieval_success_rate", result.score)
    
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
            'expected_articles': '; '.join(expected_titles),
            'predicted_articles': '; '.join(predicted_titles),
            'retrieval_success': '✓' if score == 1.0 else '✗',
            'num_expected': len(expected_article_ids),
            'num_predicted': len(predicted_article_ids),
            'overlap': len(set(expected_article_ids) & set(predicted_article_ids))
        })
    
    # Log results table
    results_df = pd.DataFrame(results_data)
    mlflow.log_table(data=results_df, artifact_file="retrieval_results.json")
    
    # Calculate additional metrics
    total_expected = sum(r['num_expected'] for r in results_data)
    total_overlap = sum(r['overlap'] for r in results_data)
    recall = total_overlap / total_expected if total_expected > 0 else 0
    
    mlflow.log_metric("article_recall", recall * 100)
    
    print("\n" + "=" * 70)
    print(f"RETRIEVAL SUCCESS RATE: {result.score:.2f}%")
    print(f"ARTICLE RECALL: {recall*100:.2f}%")
    print(f"Results logged to MLflow experiment: {full_experiment_name}")
    print("=" * 70)
    
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Retrieval Evaluation CLI")
    parser.add_argument("--retriever", choices=["title", "paragraph", "hyde", "generator"], 
                        required=True, help="Which retriever to evaluate (generator = evaluate retrieval from saved generator model)")
    parser.add_argument("--num-examples", type=int, default=50, 
                        help="Number of examples for evaluation (from the end)")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate on all examples (overrides --num-examples)")
    parser.add_argument("--max-chunks", type=int, default=None, 
                        help="Max corpus chunks to load")
    parser.add_argument("--max-articles", type=int, default=None, 
                        help="Max KB articles to load")
    parser.add_argument("--k", type=int, default=5, 
                        help="Number of passages to retrieve")
    parser.add_argument("--experiment-name", type=str, default="retrieval_evaluation",
                        help="MLflow experiment name base")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to load optimized model (HyDE or full generator)")
    parser.add_argument("--model", type=str, default="openai/gpt-4.1-mini",
                        help="LLM model to use for HyDE/generator hypothesis generation")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.retriever == "generator" and not args.load_model:
        parser.error("--load-model is required when using --retriever generator")
    
    # Include retriever type in experiment name (and optimized suffix if applicable)
    if args.load_model:
        full_experiment_name = f"{args.experiment_name}_{args.retriever}_optimized"
    else:
        full_experiment_name = f"{args.experiment_name}_{args.retriever}"
    
    # Start MLflow run at the very beginning to capture total execution time
    mlflow.set_experiment(full_experiment_name)
    with mlflow.start_run(run_name=f"{args.retriever}_k{args.k}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Configure LLM for HyDE or generator
        if args.retriever in ["hyde", "generator"]:
            print(f"Configuring LLM: {args.model}")
            lm = dspy.LM(args.model, cache=False)
            dspy.configure(lm=lm)
        
        # Load data
        train_df, kb_df, chunks_df = load_data(
            max_chunks=args.max_chunks, 
            max_articles=args.max_articles
        )
        
        # Create examples
        examples = create_examples(train_df)
        print(f"Created {len(examples)} examples")
        
        # Determine number of examples to evaluate
        if args.all:
            num_examples = len(examples)
            print(f"Evaluating on ALL {num_examples} examples")
        else:
            num_examples = args.num_examples
            print(f"Evaluating on last {num_examples} examples")
        
        # Setup retriever based on argument
        if args.retriever == "title":
            print("\n" + "=" * 70)
            print("Setting up Title Retriever...")
            print("=" * 70)
            retriever = RetrievalByTitle(kb_df=kb_df, k=args.k)
        elif args.retriever == "paragraph":
            print("\n" + "=" * 70)
            print("Setting up Paragraph Retriever...")
            print("=" * 70)
            retriever = RetrievalByParagraph(chunks_df=chunks_df, k=args.k)
        elif args.retriever == "generator":
            print("\n" + "=" * 70)
            print("Setting up Generator Retrieval Evaluation...")
            print(f"Loading generator model from: {args.load_model}")
            print("=" * 70)
            
            # Import the generator class from optimize_generator
            from optimize_generator import SimpleWixAnswerGenerator
            
            # Create the base HyDE retriever (required structure)
            hyde_retriever = HyDERetriever(
                chunks_df=chunks_df,
                kb_df=kb_df,
                num_titles=20,
                num_hypotheses=3,
                k_per_hypothesis=5,
                final_k=args.k
            )
            
            # Create and load the generator
            generator = SimpleWixAnswerGenerator(retriever=hyde_retriever)
            generator.load(args.load_model)
            print("Generator model loaded successfully!")
            
            # Wrap generator to extract only retrieval results
            retriever = GeneratorRetrievalWrapper(generator)
        else:  # hyde
            print("\n" + "=" * 70)
            print("Setting up HyDE Retriever...")
            print("=" * 70)
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
        
        # Run evaluation
        run_evaluate(
            retriever=retriever,
            examples=examples,
            kb_df=kb_df,
            num_examples=num_examples,
            retriever_type=args.retriever,
            k=args.k,
            experiment_name=args.experiment_name,
            load_model=args.load_model
        )


if __name__ == "__main__":
    main()
