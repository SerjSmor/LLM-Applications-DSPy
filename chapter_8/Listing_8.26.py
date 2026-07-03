
evaluate = Evaluate(devset=judge_test_data, num_threads=5, display_table=5,
                    display_progress=True, max_errors=3, provide_traceback=True)

evaluate(simba_qa_judge, metric=qa_judge_metric)

simba_qa_judge.save("simba_qa_judge", save_program=True)
