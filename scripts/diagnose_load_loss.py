"""Steps 1 and 5 of the base diagnosis: loading info and mean token loss (fp32) on 20 fixed English sentences."""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
H = "/mnt/e/ai-models/huggingface_cache/hub/"
S = {"base": H + "models--allenai--OLMo-2-0425-1B/snapshots/a1847dff35000b4271fa70afc5db10fd29fedbdf",
     "sft": H + "models--allenai--OLMo-2-0425-1B-SFT/snapshots/0d85a3d037876ce6ac7d4311d994400fc66ac27f",
     "ing1": H + "models--allenai--OLMo-2-0425-1B/snapshots/ada39260a92eb8cfc2252b9e937f790fd5b0dbc1"}
SENT = ["The capital of France is Paris, a city known for its museums and cafes.", "Water boils at one hundred degrees Celsius at sea level.",
 "She opened the door and found a small package waiting on the porch.", "The committee will meet on Tuesday to discuss the annual budget.",
 "Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen.", "He had lived in the village all his life and knew every path through the woods.",
 "The new software update fixes several bugs and improves battery life.", "Mount Everest is the highest mountain above sea level on Earth.",
 "In 1969, astronauts landed on the Moon for the first time in human history.", "The recipe calls for two cups of flour, one egg, and a pinch of salt.",
 "Rain fell steadily throughout the afternoon, and the streets were soon empty.", "Scientists have discovered a new species of frog in the rainforest.",
 "The train to Chicago departs from platform four at half past nine.", "Reading regularly can improve vocabulary and strengthen concentration.",
 "The river flows south through the valley before reaching the sea.", "My brother plays the guitar in a band that performs on weekends.",
 "The museum's collection includes paintings from the seventeenth century.", "Please remember to submit your report before the end of the week.",
 "A healthy diet includes fruit, vegetables, whole grains, and plenty of water.", "The old lighthouse has guided ships safely into the harbor for generations."]
tok = AutoTokenizer.from_pretrained(S["base"]); out = {}
for k, p in S.items():
    m, info = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float32, output_loading_info=True)
    m.eval(); tl = []
    with torch.no_grad():
        for s in SENT:
            ids = tok(s, return_tensors="pt").input_ids
            tl.append(float(m(ids, labels=ids).loss))
    out[k] = dict(missing_keys=list(info["missing_keys"]), unexpected_keys=list(info["unexpected_keys"]), mismatched_keys=[str(x) for x in info["mismatched_keys"]],
                  mean_loss=sum(tl) / len(tl), per_sentence=tl, tie=m.config.tie_word_embeddings)
    print(k, {x: (len(v) if isinstance(v, list) and x != "per_sentence" else v) for x, v in out[k].items() if x != "per_sentence"}, flush=True)
    del m
json.dump(out, open("data/base_diagnosis_load_loss.json", "w"), indent=1)
