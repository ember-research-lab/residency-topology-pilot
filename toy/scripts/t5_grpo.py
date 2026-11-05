"""t5: RL arm (PREREG-toy sec.1 + AMENDMENT 1). Modes (under the cap):
  gate            base pass rate on the 200 held-out items (greedy, first-integer exact match) -> toy/data/rl_gate.json; arm runs only if 5% <= rate <= 60%
  train <seed>    TRL GRPOTrainer from the base, 150 steps, wall budget 4 h -> /mnt/e/toy-residency/grpo_s<seed>, final held-out pass rate -> toy/runs/grpo_s<seed>.json
  smoke           2 steps on a tiny random Llama (API check only; no base weights)
"""
import json, re, sys, time, pathlib
import numpy as np, torch
sys.path.insert(0, "toy")
import toylib as T
from rtp import cap
cap.require_cap()
from transformers import AutoTokenizer, TrainerCallback
from datasets import Dataset
mode = sys.argv[1]
snap = json.load(open("toy/data/bases.json"))["HuggingFaceTB/SmolLM2-360M"]["snapshot"]
arith = json.load(open("toy/data/arith.json"))
tok = AutoTokenizer.from_pretrained(snap, padding_side="left")
if tok.pad_token is None: tok.pad_token = tok.eos_token
def reward_of(text, answer):
    m = re.search(r"-?\d[\d,]*", text)
    return 1.0 if m and int(m.group(0).replace(",", "")) == answer else 0.0
@torch.no_grad()
def pass_rate(model, items, bs=50):
    model.eval(); ok = 0
    for k in range(0, len(items), bs):
        b = items[k:k + bs]; enc = tok([x["prompt"] for x in b], return_tensors="pt", padding=True)
        out = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        for x, o in zip(b, out): ok += reward_of(tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True), x["answer"])
    return ok / len(items)
def vmhwm(): return int([l for l in open("/proc/self/status") if l.startswith("VmHWM")][0].split()[1])
if mode == "gate":
    r = pass_rate(T.load(snap), arith["heldout"]); out = dict(base_pass_rate=r, n=200, run_arm=bool(0.05 <= r <= 0.60))
    json.dump(out, open("toy/data/rl_gate.json", "w")); print(out); sys.exit(0)
from trl import GRPOConfig, GRPOTrainer
def rf(prompts, completions, answer, **kw): return [reward_of(c if isinstance(c, str) else c[0]["content"], a) for c, a in zip(completions, answer)]
class Budget(TrainerCallback):
    def __init__(self, s): self.s, self.t0, self.hit = s, time.time(), False
    def on_step_end(self, args, state, control, **kw):
        if time.time() - self.t0 > self.s: self.hit = True; control.should_training_stop = True
        h = [l for l in state.log_history if "reward" in l]
        if h: print("step", state.global_step, "reward", round(h[-1]["reward"], 3), "t", round(time.time() - self.t0), flush=True)
if mode == "smoke":
    from transformers import LlamaConfig, LlamaForCausalLM
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64, tie_word_embeddings=True))
    seed, steps, pbs = 0, 2, 16
else:
    seed, steps, pbs = int(sys.argv[2]), 150, 64
    assert json.load(open("toy/data/rl_gate.json"))["run_arm"], "RL gate: base pass rate outside [5%,60%]; arm not run"
    model = T.load(snap)
    torch.manual_seed(seed)
ds = Dataset.from_list([dict(prompt=x["prompt"], answer=x["answer"]) for x in arith["train"]])
cfg = GRPOConfig(output_dir=f"/mnt/e/toy-residency/grpo_tmp_{mode}", use_cpu=True, seed=seed, data_seed=seed, learning_rate=1e-6, adam_beta1=0.9, adam_beta2=0.95, weight_decay=0.0,
                 lr_scheduler_type="constant", per_device_train_batch_size=pbs, gradient_accumulation_steps=1, num_generations=8, max_completion_length=8, temperature=1.0,
                 beta=0.0, max_steps=steps, logging_steps=1, save_strategy="no", report_to="none", bf16=False, fp16=False, max_grad_norm=1.0, disable_tqdm=True)
bud = Budget(4 * 3600)
trainer = GRPOTrainer(model=model, reward_funcs=rf, args=cfg, train_dataset=ds, processing_class=tok, callbacks=[bud])
t0 = time.time(); trainer.train(); w = time.time() - t0
if mode == "smoke": print("SMOKE OK", round(w), "s"); sys.exit(0)
if bud.hit and trainer.state.global_step < steps: sys.exit(f"ARM STOPPED: 4 h wall budget hit at step {trainer.state.global_step}/{steps}")
rewards = [l["reward"] for l in trainer.state.log_history if "reward" in l]
final = pass_rate(trainer.model, arith["heldout"]); base = json.load(open("toy/data/rl_gate.json"))["base_pass_rate"]
od = pathlib.Path(f"/mnt/e/toy-residency/grpo_s{seed}"); trainer.model.save_pretrained(od)
json.dump(dict(stage="grpo", seed=seed, wall_s=round(w), vmhwm_kb=vmhwm(), steps=steps, base_pass_rate=base, final_pass_rate=final, gain_points=100 * (final - base),
               train_reward_first10=float(np.mean(rewards[:10])), train_reward_last10=float(np.mean(rewards[-10:]))), open(f"toy/runs/grpo_s{seed}.json", "w"))
print("SAVED", od, "final", final, "base", base, "wall", round(w), flush=True)
