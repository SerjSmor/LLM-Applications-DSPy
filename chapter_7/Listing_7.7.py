
from typing import Callable
from dspy import Prediction, Module

def eval_results(args, pred): 
    return (-1.0) * len(pred.short_answer.split())

class BestOfN(Module):
    def __init__(
        self,
        module: Module,
        N: int, 
        reward_fn: Callable[[dict, Prediction], float], 
        threshold: float, 
        fail_count: int | None = None, 
    ):
        self.module = module
        self.reward_fn = lambda *args: reward_fn(*args)  
        self.threshold = threshold
        self.N = N
        self.fail_count = fail_count or N  

    def forward(self, **kwargs):
        lm = self.module.get_lm() or dspy.settings.lm
        temps = [lm.kwargs["temperature"]] + [0.5 + i * (0.5 / self.N) 
                for i in range(self.N)]
        temps = list(dict.fromkeys(temps))[: self.N]
        best_pred, best_trace, best_reward = None, None, -float("inf")

        for idx, t in enumerate(temps): 
            lm_ = lm.copy(temperature=t)
            mod = self.module.deepcopy()
            mod.set_lm(lm_)

            pred = mod(**kwargs)
            reward = self.reward_fn(kwargs, pred)

            if reward > best_reward:
                best_reward, best_pred = reward, pred

            if reward >= self.threshold: 
                break

        return best_pred

base_prog = dspy.Predict("question -> short_answer")
prog = BestOfN(module=base_prog, N=3, reward_fn=eval_results, threshold=1.0)
prog(question="What is the capital of Spain?")
