
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature, ensure_signature
from typing import Any

class Closer_Example(Module):
    def __init__(self, signature: str, **config: dict[str, Any]):
        super().__init__()
        signature = ensure_signature(signature)
        signature.instructions += " Answer very carefully."
        self.predict = dspy.Predict(signature, **config) 

    def forward(self, **kwargs):
        return self.predict(**kwargs)

    async def aforward(self, **kwargs):
        return await self.predict.acall(**kwargs)

prog = Closer_Example("question -> answer")
prog(question="What is the capital of Spain?")
print(lm.inspect_history(n=1))
