
import nest_asyncio

nest_asyncio.apply()

class SkeletonOfThought(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.ChainOfThought("question -> list_of_subtasks:List[str]")
        self.predict2 = dspy.ChainOfThought("question, task -> current_answer")
        self.predict3 = dspy.ChainOfThought("question, partial_answers:List[str] ->
                                             complete_answer")

    async def aforward(self, question):
        pred = self.predict1(question=question) 
        
        subtasks_arr = []
        for task_desc in pred.list_of_subtasks: 
            subtasks_arr.append(asyncio.create_task(self.predict2.acall(question=question,
                                                                        task=task_desc)))

        answers_arr = [] 
        for subtask in subtasks_arr: 
            pred = await subtask
            answers_arr.append(pred.current_answer)

        return self.predict3(question=question, partial_answers=answers_arr) 

async def main():
    predict = SkeletonOfThought()
    business_plan = await predict.acall(
       question="Generate a strong business plan for a small accounting firm")
    print(business_plan)

asyncio.run(main())
