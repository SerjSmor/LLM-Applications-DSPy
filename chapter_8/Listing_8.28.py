evaluate = Evaluate(devset=summarizer_val_data, num_threads=5, display_table=5,
                    display_progress=True, max_errors=3, provide_traceback=True)

evaluate(prog_summarizer, metric=qa_metric)
