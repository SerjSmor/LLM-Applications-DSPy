# Building LLM Applications with DSPy

This repository contains the code examples for the book [*Building LLM Applications with DSPy*](https://www.manning.com/books/building-llm-applications-with-dspy) by Serj Smorodinsky and Brett Kennedy.

<p align="center">
  <img src="assets/1080x1080%20V1_smorodinsky.jpg" alt="Building LLM Applications with DSPy book cover" width="420">
</p>

In this repository, you will find the complete code examples for Chapters 1 through 9. They move from first predictions and signatures to evaluation, prompt optimization, custom models, summarization, and an agentic RAG-based chatbot.

- Book page: [Manning](https://www.manning.com/books/building-llm-applications-with-dspy)
- Repository contents: Chapters 1 through 9

## Table of Contents

The table below provides quick access to representative listings and the complete code folder for every chapter.

| Chapter | Topic | Main Code (Quick Access) | Full Folder |
|---|---|---|---|
| Chapter 1 | First steps with DSPy | [Listing_1_1.py](chapter_1/Listing_1_1.py) | [chapter_1](chapter_1) |
| Chapter 2 | Bootstrapping a simple intent classifier | [Listing_2.1.py](chapter_2/Listing_2.1.py), [Listing_2.8.py](chapter_2/Listing_2.8.py), [Listing_2.11.py](chapter_2/Listing_2.11.py) | [chapter_2](chapter_2) |
| Chapter 3 | Working with datasets and DSPy signatures | [Listing_3.1.py](chapter_3/Listing_3.1.py), [dspy_structures.py](chapter_3/dspy_structures.py), [openai_structures.py](chapter_3/openai_structures.py) | [chapter_3](chapter_3) |
| Chapter 4 | Evaluating classifiers and building baselines | [Listing_4.1.py](chapter_4/Listing_4.1.py), [Listing_4_5.py](chapter_4/Listing_4_5.py), [evaluate_dspy.py](chapter_4/evaluate_dspy.py), [evaluate_openai.py](chapter_4/evaluate_openai.py) | [chapter_4](chapter_4) |
| Chapter 5 | Few-shot optimization strategies | [Listing_5_1.py](chapter_5/Listing_5_1.py), [Listing_5_2.py](chapter_5/Listing_5_2.py), [optimize_few_shot_closed_form.py](chapter_5/optimize_few_shot_closed_form.py) | [chapter_5](chapter_5) |
| Chapter 6 | Instruction optimization and experiment tracking | [Listing_6.1.py](chapter_6/Listing_6.1.py), [Listing_6.2.py](chapter_6/Listing_6.2.py), [optimize_instructions_with_mlflow.py](chapter_6/optimize_instructions_with_mlflow.py) | [chapter_6](chapter_6) |
| Chapter 7 | Custom Models | [Listing_7.1.py](chapter_7/Listing_7.1.py), [Listing_7.11.py](chapter_7/Listing_7.11.py), [Listing_7.22.py](chapter_7/Listing_7.22.py) | [chapter_7](chapter_7) |
| Chapter 8 | Summarization and More Effective Metric Functions | [Listing_8.1.py](chapter_8/Listing_8.1.py), [Listing_8.14.py](chapter_8/Listing_8.14.py), [Listing_8.28.py](chapter_8/Listing_8.28.py) | [chapter_8](chapter_8) |
| Chapter 9 | Creating an Agentic RAG-Based Chatbot | [Listing_9.1.py](chapter_9/Listing_9.1.py), [Listing_9.8.py](chapter_9/Listing_9.8.py), [Listing_9.16.py](chapter_9/Listing_9.16.py) | [chapter_9](chapter_9) |

## Repository Overview

The code examples use DSPy to build and improve LLM programs, beginning with a practical intent-classification workflow and progressing to custom model integrations and agentic retrieval. Across the nine chapters, the repository covers:

- defining DSPy programs and signatures
- creating examples from the ATIS dataset
- evaluating baseline programs
- optimizing prompts with labeled few-shot and bootstrapping methods
- comparing optimization strategies such as `BootstrapFewShot`, `COPRO`, `MIPROv2`, `GEPA`, and `SIMBA`
- tracking optimization runs with MLflow
- integrating custom models with DSPy
- building summarization programs and more effective metric functions
- calibrating an LLM as a judge for dialogue summaries
- creating an agentic RAG-based chatbot

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

Some examples also depend on external services or libraries such as Hugging Face datasets, sentence-transformers, MLflow, and vector databases. Check the relevant chapter code for any chapter-specific configuration.

## Notes

- The repository contains the code for all nine chapters.
- Each chapter folder includes its numbered listings and any supporting notebooks, data, or configuration files.
- Several examples use the ATIS dataset and OpenAI-hosted models such as `gpt-4o-mini` and `gpt-4.1-nano`.
