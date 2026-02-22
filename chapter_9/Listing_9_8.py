from datetime import datetime

import dspy
import mlflow
import pandas as pd

from chapter_9.listing_9_4 import load_data, RetrievalByTitle
from chapter_9.Listing_9_7 import create_examples


def retrieval_success_at_k(example: dspy.Example, prediction: dspy.Prediction, trace=None):
    """Metric that calculates the percentage of ground-truth articles retrieved.
    
    Returns the recall: (number of correct articles retrieved) / (total expected articles).
    This is an objective metric requiring no LLM judge.
    """
    expected = set(example.article_ids)
    predicted = set(prediction.article_ids)
    num_correct = len(expected.intersection(predicted))
    return num_correct / len(expected) if len(expected) > 0 else 0.0


def run_retrieval_experiment(retriever, test_examples, retriever_name, k, embedder_name):
    """Run a single retrieval evaluation experiment and log results to MLflow."""
    run_name = f"{retriever_name}_k{k}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("retriever", retriever_name)
        mlflow.log_param("k", k)
        mlflow.log_param("embedder", embedder_name)
        mlflow.log_param("num_examples", len(test_examples))
        mlflow.log_param("metric", "retrieval_success_at_k")

        print("\n" + "=" * 70)
        print(f"EXPERIMENT: retriever={retriever_name}, k={k}, embedder={embedder_name}")
        print("=" * 70)

        start_time = datetime.now()
        print(f"Starting at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        evaluator = dspy.Evaluate(
            devset=test_examples,
            num_threads=4,
            display_progress=True,
            display_table=5,
            max_errors=10
        )

        result = evaluator(retriever, metric=retrieval_success_at_k)

        duration = datetime.now() - start_time
        print(f"Duration: {duration}")

        mlflow.log_metric("retrieval_success_at_k", result.score)

        print(f"SCORE: {result.score:.2f}%")
        print("=" * 70)
        return result


if __name__ == "__main__":
    # Load data (once)
    train_df, kb_df = load_data()
    examples = create_examples(train_df)
    print(f"Created {len(examples)} examples")

    # Test set: last 200 examples (matches notebook evaluation size)
    num_examples = 200
    test_examples = examples[-num_examples:]
    print(f"Test set: {len(test_examples)} examples")

    # Embedder configuration
    embedder_name = 'BAAI/bge-small-en-v1.5'

    # Set MLflow experiment
    mlflow.set_experiment("chapter_9_retrieval_baseline_new")

    # Evaluate title retriever at different k values
    k_values = [15, 20]
    
    for k in k_values:
        print(f"\n{'=' * 70}")
        print(f"Building title retriever with k={k}...")
        print(f"{'=' * 70}")
        
        # Build retriever (expensive -- embeds 6K titles)
        retriever = RetrievalByTitle(kb_df=kb_df, embedder_name=embedder_name, k=k)
        
        # Run experiment
        run_retrieval_experiment(
            retriever=retriever,
            test_examples=test_examples,
            retriever_name="title",
            k=k,
            embedder_name=embedder_name
        )
