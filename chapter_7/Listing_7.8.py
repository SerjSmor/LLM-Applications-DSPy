
from dspy import InputField, OutputField, Predict
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature, ensure_signature

class MultiChainComparison(Module):
    def __init__(self, signature, M=3, temperature=0.7, **config):  
        super().__init__()

        self.M = M
        signature = ensure_signature(signature) 

        *_, self.last_key = signature.output_fields.keys() 

        for idx in range(M): 
            signature = signature.append(
                f"reasoning_attempt_{idx+1}",
                InputField(desc="${reasoning attempt}"),
            )

        signature = signature.prepend( 
            "Rationale", OutputField(desc="${corrected reasoning}"),)

        self.predict = Predict(signature, temperature=temperature, **config) 

    def forward(self, completions, **kwargs):
        attempts = [] 

        for c in completions: 
            rationale = c.get("rationale", c.get("reasoning")).strip().split("\n")[0]
            answer = str(c[self.last_key]).strip().split("\n")[0].strip()
            attempts.append(
                f"«I'm trying to {rationale} I'm not sure but my prediction is {answer}»",
            )

        kwargs = {
            **{f"reasoning_attempt_{idx+1}": attempt for idx, attempt in
                  enumerate(attempts)},
            **kwargs,
        }
        return self.predict(**kwargs) 

lm = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY, cache=False)
dspy.settings.configure(lm=lm)

qa_signature = "question -> answer"
base_prog = dspy.ChainOfThought(qa_signature)
question = "What is a good way to find an apartment?"
pred1 = base_prog(question=question)
pred2 = base_prog(question=question)
pred3 = base_prog(question=question)
prog = MultiChainComparison(signature=qa_signature, M=3)
prog(completions=[pred1, pred2, pred3], question=question)
