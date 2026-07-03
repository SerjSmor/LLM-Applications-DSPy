import dspy
import asyncio
import os
import time

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False))

class MyModule(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.Predict("question->answer") 

    async def aforward(self, question, **kwargs):
        print(question) 
        await asyncio.sleep(5) 
        pred = await self.predict1.acall(question=question) 
        pred['Details'] = "From async forward method" 
        return pred

    def forward(self, question):
        pred = self.predict1(question=question)
        pred['Details'] = "From regular forward method"
        return pred

async def main():
    start_time = time.time()
    task_list = asyncio.gather(myModule.acall(question="What is the capital of Finland?"),
                               myModule.acall(question="What is the capital of Norway?"))
    answer1, answer2 = await task_list 
    print(answer1)
    print(answer2)
    end_time = time.time()
    print("time", end_time-start_time)

myModule= MyModule()
print(myModule(question="What is the capital of Brazil?"))
asyncio.run(main())

