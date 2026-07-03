import dspy
import asyncio
import os

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
predict = dspy.Predict("question->answer")

async def main():
    task_list = asyncio.gather(predict.acall(question="What is the capital of Finland?"),
                               predict.acall(question="What is the capital of Norway?")) 
    answer1, answer2 = await task_list 
    print(answer1)
    print(answer2)

asyncio.run(main())
