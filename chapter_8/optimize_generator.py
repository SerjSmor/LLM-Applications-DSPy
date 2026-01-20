#!/home/serj/dev/prompt_programming_with_dspy/venv/bin/python
"""
Generator Optimization Script

Optimizes the SimpleWixAnswerGenerator using MIPROv2 with two modes:
1. Joint mode: Optimize HyDE retriever + answer generator together
2. Frozen mode: Use pre-optimized HyDE (frozen), only optimize the answer generator

After optimization completes, automatically runs evaluation on the test set.

Usage:
    # Joint optimization (optimize both HyDE + Generator)
    python chapter_8/optimize_generator.py --mode joint --train-size 50 --val-size 25 --test-size 50

    # Frozen HyDE optimization (only optimize generator)
    python chapter_8/optimize_generator.py --mode frozen --load-hyde chapter_8/optimized_mipro_hyde.pkl --test-size 50

    # Standalone evaluation
    python chapter_8/optimize_generator.py --mode evaluate --load-model chapter_8/optimized_joint_generator.pkl --test-size 50
"""

import argparse
from datetime import datetime

import datasets
import dspy
import mlflow
import pandas as pd
from dspy.evaluate import SemanticF1

from retrieval import HyDERetriever


# =============================================================================
# Data Loading (reused from simple_generator.py)
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
# DSPy Signatures
# =============================================================================

class AnswerGenerator(dspy.Signature):
    """Generate a helpful answer to the user's question based on retrieved KB articles."""
    question: str = dspy.InputField(desc="User's question about Wix")
    relevant_articles: list[str] = dspy.InputField(desc="Retrieved KB article passages")
    answer: str = dspy.OutputField(desc="Helpful answer based on the retrieved articles")


# =============================================================================
# Answer Generator Module
# =============================================================================

class SimpleWixAnswerGenerator(dspy.Module):
    """End-to-end answer generator with HyDE retriever.
    
    Contains both the HyDE retriever and answer generator as DSPy modules.
    In joint mode, MIPROv2 optimizes both. In frozen mode, set retriever._compiled = True
    to freeze it, and MIPROv2 will only optimize the answer_generator.
    """
    
    def __init__(self, retriever: HyDERetriever):
        self.retriever = retriever
        self.answer_generator = dspy.ChainOfThought(AnswerGenerator)

    def forward(self, question: str):
        # Step 1: Retrieve relevant articles using HyDE
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

def create_semantic_f1_metric():
    """Create the DSPy SemanticF1 metric wrapper."""
    semantic_f1 = SemanticF1()
    
    def metric(example: dspy.Example, prediction: dspy.Prediction, trace=None):
        """Wrapper metric that adapts our field names to SemanticF1's expected format."""
        adapted_example = dspy.Example(
            question=example.question,
            response=example.answer
        )
        adapted_pred = dspy.Prediction(
            response=prediction.answer
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


def retrieval_success_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None):
    """Metric: checks if all expected articles were retrieved."""
    expected = set(example.article_ids)
    predicted = set(prediction.article_ids) if prediction.article_ids else set()
    return 1.0 if expected.issubset(predicted) else 0.0


# =============================================================================
# Evaluation
# =============================================================================

def run_evaluation(generator, test_examples: list[dspy.Example], 
                   experiment_name: str, run_name: str):
    """Run evaluation on test set with MLflow logging.
    
    Reports both SemanticF1 (answer quality) and Retrieval Success (retrieval quality).
    """
    print("\n" + "=" * 70)
    print(f"EVALUATION ON TEST SET - {len(test_examples)} examples")
    print("=" * 70)
    
    semantic_f1_metric = create_semantic_f1_metric()
    
    # Run SemanticF1 evaluation
    print(f"\nStarting evaluation at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = datetime.now()
    
    print("\nEvaluating SemanticF1...")
    evaluator = dspy.Evaluate(
        devset=test_examples,
        num_threads=4,
        display_progress=True,
        max_errors=10
    )
    semantic_f1_result = evaluator(generator, metric=semantic_f1_metric)
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Evaluation finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Evaluation duration: {duration}")
    
    # Calculate retrieval success from the same results
    print("\nCalculating Retrieval Success...")
    retrieval_successes = 0
    for example, prediction, _ in semantic_f1_result.results:
        if retrieval_success_metric(example, prediction) == 1.0:
            retrieval_successes += 1
    
    retrieval_success_rate = (retrieval_successes / len(test_examples)) * 100
    
    # Log metrics to current MLflow run
    mlflow.log_metric("test_semantic_f1", semantic_f1_result.score)
    mlflow.log_metric("test_retrieval_success", retrieval_success_rate)
    
    print("\n" + "=" * 70)
    print(f"TEST SET RESULTS:")
    print(f"  SemanticF1:        {semantic_f1_result.score:.2f}%")
    print(f"  Retrieval Success: {retrieval_success_rate:.2f}% ({retrieval_successes}/{len(test_examples)})")
    print("=" * 70)
    
    return semantic_f1_result, retrieval_success_rate


# =============================================================================
# Joint Optimization Mode
# =============================================================================

def run_joint_optimization(chunks_df: pd.DataFrame, kb_df: pd.DataFrame,
                           train_examples: list[dspy.Example],
                           val_examples: list[dspy.Example],
                           test_examples: list[dspy.Example],
                           output_path: str,
                           mipro_mode: str = "light",
                           max_bootstrapped_demos: int = 3,
                           max_labeled_demos: int = 4,
                           k: int = 5):
    """Run joint optimization of HyDE + Generator using MIPROv2."""
    print("\n" + "=" * 70)
    print("JOINT OPTIMIZATION MODE")
    print("=" * 70)
    print(f"Train set: {len(train_examples)} examples")
    print(f"Val set: {len(val_examples)} examples")
    print(f"Test set: {len(test_examples)} examples")
    print(f"MIPROv2 mode: {mipro_mode}")
    print(f"max_bootstrapped_demos: {max_bootstrapped_demos}")
    print(f"max_labeled_demos: {max_labeled_demos}")
    print("=" * 70)
    
    # Create HyDE retriever
    print("\nSetting up HyDE Retriever...")
    hyde_retriever = HyDERetriever(
        chunks_df=chunks_df,
        kb_df=kb_df,
        num_titles=20,
        num_hypotheses=3,
        k_per_hypothesis=5,
        final_k=k
    )
    
    # Create full generator (HyDE + answer generator)
    generator = SimpleWixAnswerGenerator(retriever=hyde_retriever)
    
    # Create metric
    metric = create_semantic_f1_metric()
    
    # Run MIPROv2 optimization
    print(f"\nStarting optimization at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    opt_start_time = datetime.now()
    
    print("\nStarting MIPROv2 optimization...")
    mipro = dspy.MIPROv2(metric=metric, auto=mipro_mode)
    
    optimized_generator = mipro.compile(
        generator,
        trainset=train_examples,
        valset=val_examples,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        requires_permission_to_run=False
    )
    
    opt_end_time = datetime.now()
    opt_duration = opt_end_time - opt_start_time
    print(f"Optimization finished at: {opt_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Optimization duration: {opt_duration}")
    
    print("\nOptimization complete!")
    
    # Save the optimized model
    optimized_generator.save(output_path)
    print(f"Saved optimized model to: {output_path}")
    
    # Run evaluation on test set
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    semantic_f1_result, retrieval_success = run_evaluation(
        generator=optimized_generator,
        test_examples=test_examples,
        experiment_name="generator_optimization_joint",
        run_name=f"joint_test_eval_{timestamp}"
    )
    
    return optimized_generator, semantic_f1_result, retrieval_success


# =============================================================================
# Frozen HyDE Optimization Mode
# =============================================================================

def run_frozen_optimization(chunks_df: pd.DataFrame, kb_df: pd.DataFrame,
                            train_examples: list[dspy.Example],
                            val_examples: list[dspy.Example],
                            test_examples: list[dspy.Example],
                            hyde_model_path: str,
                            output_path: str,
                            mipro_mode: str = "light",
                            max_bootstrapped_demos: int = 3,
                            max_labeled_demos: int = 4,
                            k: int = 5):
    """Run optimization with frozen HyDE - only optimize the answer generator."""
    print("\n" + "=" * 70)
    print("FROZEN HYDE OPTIMIZATION MODE")
    print("=" * 70)
    print(f"Train set: {len(train_examples)} examples")
    print(f"Val set: {len(val_examples)} examples")
    print(f"Test set: {len(test_examples)} examples")
    print(f"Frozen HyDE model: {hyde_model_path}")
    print(f"MIPROv2 mode: {mipro_mode}")
    print(f"max_bootstrapped_demos: {max_bootstrapped_demos}")
    print(f"max_labeled_demos: {max_labeled_demos}")
    print("=" * 70)
    
    # Create and load optimized HyDE retriever
    print("\nLoading pre-optimized HyDE Retriever...")
    hyde_retriever = HyDERetriever(
        chunks_df=chunks_df,
        kb_df=kb_df,
        num_titles=20,
        num_hypotheses=3,
        k_per_hypothesis=5,
        final_k=k
    )
    hyde_retriever.load(hyde_model_path)
    hyde_retriever._compiled = True  # Freeze the retriever - MIPROv2 won't optimize it
    print("Loaded and froze HyDE successfully!")
    
    # Create generator with frozen retriever (same as joint mode)
    generator = SimpleWixAnswerGenerator(retriever=hyde_retriever)
    
    # Create metric (same as joint mode)
    metric = create_semantic_f1_metric()
    
    # Run MIPROv2 optimization - will only optimize answer_generator (not frozen retriever)
    print(f"\nStarting optimization at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    opt_start_time = datetime.now()
    
    print("\nStarting MIPROv2 optimization (answer generator only, HyDE frozen)...")
    mipro = dspy.MIPROv2(metric=metric, auto=mipro_mode)
    
    optimized_generator = mipro.compile(
        generator,
        trainset=train_examples,
        valset=val_examples,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        requires_permission_to_run=False
    )
    
    opt_end_time = datetime.now()
    opt_duration = opt_end_time - opt_start_time
    print(f"Optimization finished at: {opt_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Optimization duration: {opt_duration}")
    
    print("\nOptimization complete!")
    
    # Save the optimized model
    optimized_generator.save(output_path)
    print(f"Saved optimized generator to: {output_path}")
    
    # Run evaluation on test set
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    semantic_f1_result, retrieval_success = run_evaluation(
        generator=optimized_generator,
        test_examples=test_examples,
        experiment_name="generator_optimization_frozen",
        run_name=f"frozen_test_eval_{timestamp}"
    )
    
    return optimized_generator, semantic_f1_result, retrieval_success


# =============================================================================
# Standalone Evaluation Mode
# =============================================================================

def run_standalone_evaluation(chunks_df: pd.DataFrame, kb_df: pd.DataFrame,
                              test_examples: list[dspy.Example],
                              model_path: str,
                              hyde_model_path: str = None,
                              mode: str = "joint",
                              k: int = 5):
    """Run standalone evaluation on a saved model."""
    print("\n" + "=" * 70)
    print("STANDALONE EVALUATION MODE")
    print("=" * 70)
    print(f"Test set: {len(test_examples)} examples")
    print(f"Model path: {model_path}")
    print(f"Mode: {mode}")
    if hyde_model_path:
        print(f"HyDE model path: {hyde_model_path}")
    print("=" * 70)
    
    # Create HyDE retriever
    print("\nSetting up HyDE Retriever...")
    hyde_retriever = HyDERetriever(
        chunks_df=chunks_df,
        kb_df=kb_df,
        num_titles=20,
        num_hypotheses=3,
        k_per_hypothesis=5,
        final_k=k
    )
    
    # For frozen mode, load HyDE state separately and freeze it
    if mode == "frozen":
        if not hyde_model_path:
            raise ValueError("--load-hyde is required for frozen mode evaluation")
        hyde_retriever.load(hyde_model_path)
        hyde_retriever._compiled = True
        print("Loaded and froze HyDE retriever!")
    
    # Create and load generator (same structure for both modes)
    generator = SimpleWixAnswerGenerator(retriever=hyde_retriever)
    generator.load(model_path)
    print(f"Loaded {mode} model successfully!")
    
    # Run evaluation
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    semantic_f1_result, retrieval_success = run_evaluation(
        generator=generator,
        test_examples=test_examples,
        experiment_name=f"generator_evaluation_{mode}",
        run_name=f"{mode}_eval_{timestamp}"
    )
    
    return semantic_f1_result, retrieval_success


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generator Optimization CLI")
    parser.add_argument("--mode", choices=["joint", "frozen", "evaluate"], 
                        required=True, help="Optimization mode: joint (HyDE+Generator), frozen (Generator only), or evaluate")
    parser.add_argument("--train-size", type=int, default=50, 
                        help="Number of training examples")
    parser.add_argument("--val-size", type=int, default=25, 
                        help="Number of validation examples")
    parser.add_argument("--test-size", type=int, default=50, 
                        help="Number of test examples (from the end)")
    parser.add_argument("--max-chunks", type=int, default=None, 
                        help="Max corpus chunks to load")
    parser.add_argument("--max-articles", type=int, default=None, 
                        help="Max KB articles to load")
    parser.add_argument("--model", type=str, default="openai/gpt-4.1-mini", 
                        help="LLM model to use")
    parser.add_argument("--k", type=int, default=5, 
                        help="Number of passages to retrieve")
    parser.add_argument("--load-hyde", type=str, default=None,
                        help="Path to pre-optimized HyDE model (required for frozen mode)")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to load optimized model for evaluation")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for optimized model")
    parser.add_argument("--mipro-mode", type=str, default="light",
                        choices=["light", "medium", "heavy"],
                        help="MIPROv2 optimization mode")
    parser.add_argument("--max-bootstrapped", type=int, default=3,
                        help="Max bootstrapped demos for MIPROv2")
    parser.add_argument("--max-labeled", type=int, default=4,
                        help="Max labeled demos for MIPROv2")
    
    args = parser.parse_args()
    
    # Set default output paths based on mode
    if args.output is None:
        if args.mode == "joint":
            args.output = "chapter_8/optimized_joint_generator.pkl"
        elif args.mode == "frozen":
            args.output = "chapter_8/optimized_generator_frozen_hyde.pkl"
    
    # Validate arguments
    if args.mode == "frozen" and not args.load_hyde:
        parser.error("--load-hyde is required for frozen mode")
    if args.mode == "evaluate" and not args.load_model:
        parser.error("--load-model is required for evaluate mode")
    
    # Configure LLM
    print(f"Configuring LLM: {args.model}")
    lm = dspy.LM(args.model, cache=False)
    dspy.configure(lm=lm)
    
    # Start MLflow run at the very beginning to capture total execution time
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.mode == "joint":
        experiment_name = "generator_optimization_joint"
        run_name = f"joint_opt_{timestamp}"
    elif args.mode == "frozen":
        experiment_name = "generator_optimization_frozen"
        run_name = f"frozen_opt_{timestamp}"
    else:  # evaluate
        # Determine if this is a joint or frozen model based on hyde flag
        eval_mode = "frozen" if args.load_hyde else "joint"
        experiment_name = f"generator_evaluation_{eval_mode}"
        run_name = f"{eval_mode}_eval_{timestamp}"
    
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        # Log common parameters
        mlflow.log_param("mode", args.mode)
        mlflow.log_param("model", args.model)
        mlflow.log_param("k", args.k)
        if args.mode != "evaluate":
            mlflow.log_param("train_size", args.train_size)
            mlflow.log_param("val_size", args.val_size)
            mlflow.log_param("mipro_mode", args.mipro_mode)

        # Load data
        train_df, kb_df, chunks_df = load_data(
            max_chunks=args.max_chunks, 
            max_articles=args.max_articles
        )
        
        # Create examples
        all_examples = create_examples(train_df)
        print(f"Created {len(all_examples)} total examples")
        
        # Split into train/val/test
        test_examples = all_examples[-args.test_size:]
        remaining = all_examples[:-args.test_size]
        
        if args.mode != "evaluate":
            train_examples = remaining[:args.train_size]
            val_examples = remaining[args.train_size:args.train_size + args.val_size]
            
            print(f"\nData split:")
            print(f"  Train: {len(train_examples)} examples (indices 0-{args.train_size-1})")
            print(f"  Val: {len(val_examples)} examples (indices {args.train_size}-{args.train_size + args.val_size - 1})")
            print(f"  Test: {len(test_examples)} examples (last {args.test_size})")
        
        # Run the appropriate mode
        if args.mode == "joint":
            run_joint_optimization(
                chunks_df=chunks_df,
                kb_df=kb_df,
                train_examples=train_examples,
                val_examples=val_examples,
                test_examples=test_examples,
                output_path=args.output,
                mipro_mode=args.mipro_mode,
                max_bootstrapped_demos=args.max_bootstrapped,
                max_labeled_demos=args.max_labeled,
                k=args.k
            )
        elif args.mode == "frozen":
            run_frozen_optimization(
                chunks_df=chunks_df,
                kb_df=kb_df,
                train_examples=train_examples,
                val_examples=val_examples,
                test_examples=test_examples,
                hyde_model_path=args.load_hyde,
                output_path=args.output,
                mipro_mode=args.mipro_mode,
                max_bootstrapped_demos=args.max_bootstrapped,
                max_labeled_demos=args.max_labeled,
                k=args.k
            )
        elif args.mode == "evaluate":
            # Determine if this is a joint or frozen model based on hyde flag
            eval_mode = "frozen" if args.load_hyde else "joint"
            run_standalone_evaluation(
                chunks_df=chunks_df,
                kb_df=kb_df,
                test_examples=test_examples,
                model_path=args.load_model,
                hyde_model_path=args.load_hyde,
                mode=eval_mode,
                k=args.k
            )


if __name__ == "__main__":
    main()
