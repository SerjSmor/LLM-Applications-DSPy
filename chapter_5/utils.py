import dspy
import mlflow
import pandas as pd
from dspy import Evaluate

from datasets import load_dataset
from chapter_4.evaluate_dspy import validate_answer
from common.consts import ATIS_INTENT_MAPPING


def evaluate_model(examples: list[dspy.Example], lm: dspy.LM, classifier: dspy.Module, is_mlflow=False, optimizer="",
                   **mlflow_logs):
    import time
    dspy.settings.configure(lm=lm)
    evaluate_atis = Evaluate(devset=examples, num_threads=10, display_progress=True, display_table=5,
                             max_errors=10000,
                             provide_traceback=True)
    start_time = time.time()
    results = evaluate_atis(classifier, metric=validate_answer)
    end_time = time.time()
    if is_mlflow:
        classifier_name = classifier.__class__.__name__
        with mlflow.start_run(run_name=f"{optimizer}_{classifier_name}_n{len(examples)}"):
            mlflow.log_metric("exact_match", results.score)
            mlflow.log_param("optimizer", optimizer)
            mlflow.log_param("module", classifier_name)
            mlflow.log_param("lm", lm.model)
            mlflow.log_param("num_examples", len(examples))
            if len(mlflow_logs) > 0:
                for key, value in mlflow_logs.items():
                    mlflow.log_param(key, value)
            mlflow.end_run()

    print(f"total time: {end_time - start_time}")

def create_examples_from_set(set_name, n):
    ds = load_dataset("tuetschek/atis")
    ds.set_format(type='pandas')
    df: pd.DataFrame = ds[set_name][:]
    df['intent'] = df['intent'].map(ATIS_INTENT_MAPPING)
    df = df.dropna(subset='intent')
    if n > 0:
        df = df.sample(n=n, random_state=42)  # A

    examples = []
    for index in df.index:  # B
        row = df.loc[index]
        examples.append(
            dspy.Example(message=row['text'],
                         intent_label=row['intent']).with_inputs('message')
        )
    return examples