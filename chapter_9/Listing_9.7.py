from datetime import datetime

import dspy
import mlflow
mlflow.dspy.autolog(log_traces_from_compile=True)
import pandas as pd

from chapter_9.listing_9_4 import load_data, RetrievalByTitle
from chapter_9.listing_9_5 import RAG
from chapter_9.listing_9_6 import FactualAlignmentJudge, LLMJudge


class EmptyRetriever(dspy.Module):
    """Stub retriever that returns nothing -- used as a no-retrieval baseline."""
    def forward(self, question: str):
        return dspy.Prediction(passages=[], article_ids=[])


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


def run_experiment(generator, test_examples, llm_judge,
                   model_name, judge_model_name, retriever_type, k, num_examples):
    """Run a single evaluation experiment and log results to MLflow."""
    run_name = f"{retriever_type}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("num_examples", num_examples)
        mlflow.log_param("metric", "factual_alignment")
        mlflow.log_param("generator_model", model_name)
        mlflow.log_param("judge_model", judge_model_name)
        mlflow.log_param("k", k)
        mlflow.log_param("retriever", retriever_type)

        print("\n" + "=" * 70)
        print(f"EXPERIMENT: retriever={retriever_type}, generator={model_name}, judge={judge_model_name}")
        print("=" * 70)

        start_time = datetime.now()
        print(f"Starting at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        evaluator = dspy.Evaluate(
            devset=test_examples,
            num_threads=4,
            display_progress=True,
            max_errors=10
        )

        result = evaluator(generator, metric=llm_judge)

        duration = datetime.now() - start_time
        print(f"Duration: {duration}")

        mlflow.log_metric("factual_alignment_score", result.score)

        print(f"SCORE: {result.score:.2f}%")
        print("=" * 70)
        return result


if __name__ == "__main__":
    # Load data (once)
    train_df, kb_df = load_data()
    examples = create_examples(train_df)
    print(f"Created {len(examples)} examples")

    # Test set: last 50 examples
    k = 5
    num_examples = 50
    test_examples = examples[-num_examples:]
    print(f"Test set: {len(test_examples)} examples")

    # Build title retriever once (expensive -- embeds 6K titles)
    print("\nBuilding title retriever...")
    title_retriever = RetrievalByTitle(kb_df=kb_df, k=k)

    # Define experiments
    experiments = [
        {"model": "openai/gpt-4.1-mini", "retriever_type": "title"},
        {"model": "openai/gpt-4.1-mini", "retriever_type": "empty"},
        {"model": "openai/gpt-4.1-nano", "retriever_type": "title"},
        {"model": "openai/gpt-4.1-nano", "retriever_type": "empty"},
    ]

    # Judge LM: pinned to a strong model so evaluation is consistent across experiments
    judge_model_name = "openai/gpt-4.1-mini"
    judge_lm = dspy.LM(judge_model_name, cache=False)

    mlflow.set_experiment("chapter_9_baseline_stricter_instructions")

    for exp in experiments:
        # Configure generator LLM for this experiment
        gen_lm = dspy.LM(exp["model"], cache=False)
        dspy.configure(lm=gen_lm)

        # Pick retriever
        if exp["retriever_type"] == "title":
            retriever = title_retriever
        else:
            retriever = EmptyRetriever()

        generator = RAG(retriever=retriever, kb_df=kb_df)

        # Judge predictor pinned to judge_lm (independent of global config)
        predictor = dspy.ChainOfThought(FactualAlignmentJudge)
        predictor.lm = judge_lm
        llm_judge = LLMJudge(predictor)

        run_experiment(
            generator=generator,
            test_examples=test_examples,
            llm_judge=llm_judge,
            model_name=exp["model"],
            judge_model_name=judge_model_name,
            retriever_type=exp["retriever_type"],
            k=k,
            num_examples=num_examples
        )
