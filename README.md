# Building LLM Applications with DSPy

This repository contains the code examples for the book [*Building LLM Applications with DSPy*](https://www.manning.com/books/building-llm-applications-with-dspy) by Serj Smorodinsky and Brett Kennedy.

<p align="center">
  <img src="assets/1080x1080%20V1_smorodinsky.jpg" alt="Building LLM Applications with DSPy book cover" width="420">
</p>

In this repository, you will find practical DSPy examples that move from first predictions and signatures to evaluation, few-shot optimization, and instruction optimization workflows.

- Book page: [Manning](https://www.manning.com/books/building-llm-applications-with-dspy)
- Repository focus in this README: Chapters 1 to 6

## Table of Contents

This README currently focuses on the first six chapters and follows a quick-access layout similar to the structure used in the reference repository.

| Chapter | Topic | Main Code (Quick Access) | Full Folder |
|---|---|---|---|
| Chapter 1 | First steps with DSPy | [Listing_1_1.py](chapter_1/Listing_1_1.py) | [chapter_1](chapter_1) |
| Chapter 2 | Bootstrapping a simple intent classifier | [Listing_2.1.py](chapter_2/Listing_2.1.py), [Listing_2.8.py](chapter_2/Listing_2.8.py), [Listing_2.11.py](chapter_2/Listing_2.11.py) | [chapter_2](chapter_2) |
| Chapter 3 | Working with datasets and DSPy signatures | [Listing_3.1.py](chapter_3/Listing_3.1.py), [dspy_structures.py](chapter_3/dspy_structures.py), [openai_structures.py](chapter_3/openai_structures.py) | [chapter_3](chapter_3) |
| Chapter 4 | Evaluating classifiers and building baselines | [Listing_4.1.py](chapter_4/Listing_4.1.py), [Listing_4_5.py](chapter_4/Listing_4_5.py), [evaluate_dspy.py](chapter_4/evaluate_dspy.py), [evaluate_openai.py](chapter_4/evaluate_openai.py) | [chapter_4](chapter_4) |
| Chapter 5 | Few-shot optimization strategies | [Listing_5_1.py](chapter_5/Listing_5_1.py), [Listing_5_2.py](chapter_5/Listing_5_2.py), [optimize_few_shot_closed_form.py](chapter_5/optimize_few_shot_closed_form.py) | [chapter_5](chapter_5) |
| Chapter 6 | Instruction optimization and experiment tracking | [Listing_6.1.py](chapter_6/Listing_6.1.py), [Listing_6.2.py](chapter_6/Listing_6.2.py), [optimize_instructions_with_mlflow.py](chapter_6/optimize_instructions_with_mlflow.py) | [chapter_6](chapter_6) |

## Repository Overview

The code examples use DSPy to build and improve LLM programs on top of a practical intent-classification workflow. Across the first six chapters, the repository covers:

- defining DSPy programs and signatures
- creating examples from the ATIS dataset
- evaluating baseline programs
- optimizing prompts with labeled few-shot and bootstrapping methods
- comparing optimization strategies such as `BootstrapFewShot`, `COPRO`, and `MIPROv2`
- tracking optimization runs with MLflow

## Setup

Python 3.11 or newer is recommended.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you prefer `uv`, the equivalent install flow is:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Many examples expect model credentials in your environment, for example:

```bash
export OPENAI_API_KEY=your_key_here
```

Some later examples in these chapters also depend on external services or libraries such as Hugging Face datasets, sentence-transformers, and MLflow.

## Notes

- The repository already contains additional material beyond Chapter 6.
- This README intentionally documents only the first six chapters for now.
- Several examples use the ATIS dataset and OpenAI-hosted models such as `gpt-4o-mini` and `gpt-4.1-nano`.
