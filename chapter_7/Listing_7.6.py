
class SimplerChainOfThought(Module):
    def __init__(
        self,
        signature: str | type[Signature], 
        **config: dict[str, Any],
    ):
        super().__init__()
        signature = ensure_signature(signature) 
        prefix = "Reasoning: Let's think step by step in order to" 
        desc = "${reasoning}" 
        rationale_field = dspy.OutputField(prefix=prefix, desc=desc)
        extended_signature = signature.prepend(name="reasoning", field=rationale_field) 
        print(extended_signature.fields) 
        self.predict = dspy.Predict(extended_signature, **config) 

    def forward(self, **kwargs):
        return self.predict(**kwargs)

    async def aforward(self, **kwargs):
        return await self.predict.acall(**kwargs)

prog = SimplerChainOfThought("question -> answer")
prog(question="What is the capital of Spain?")
