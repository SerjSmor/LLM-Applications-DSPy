
import dspy
import asyncio
import os

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False))

class MyModule(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.ChainOfThought("question -> answer")
        self.predict2 = dspy.ChainOfThought("answers:List[str] -> best_answer")

    async def aforward(self, question, **kwargs):
        task_list = asyncio.gather(self.predict1.acall(question=question),
                                   self.predict1.acall(question=question))
        answer1, answer2 = await task_list
        pred = self.predict2(answers=[answer1.answer, answer2.answer])
        return pred

async def main():
    mod = MyModule()
    result = await mod.acall(question="When is a good time to buy a house?")
    print(result) 

asyncio.run(main())
