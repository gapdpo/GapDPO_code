# GapDPO: Setwise Direct Preference Optimization for LLM Safety Alignment

Code and data for the experiments of the paper. GapDPO is implemented inside the short-length
adversarial training framework of Adv-ICL (https://github.com/fshp971/adv-icl): the framework
(GCG adversary in the training loop, Alpaca utility term, LoRA) is unchanged and only the alignment
loss L_align is replaced.

## Layout

```
src/
  train_gapdpo.py        GapDPO: (M, K) sets, mean-gap loss + alpha_c * V_c + alpha_h * V_h
  train_gapdpo_m4k1.py   GapDPO (M=4, K=1) run of Table 2
  train_dpo.py           pairwise DPO baseline
  train_sft.py           SFT baseline (safe-redirect style, cross entropy)
  train_adv_icl.py       Adv-ICL baseline [27]
  dpo_common.py, dpo_listwise.py, chosen4_data.py, chosen4_fixed_data.py   shared DPO trainer and data code
  adv_trainers/          training loop, GCG-in-the-loop, utility term, DPO trainers, evaluator/judge
  attacks/               GCG, PAIR, ... (attack implementations of [27])
  utils/                 datasets, model loading, argument parsing
  evaluate.py            builds attacked eval sets (GCG / PAIR) and runs the HarmBench judge
  evaluate_amplegcg.py   AmpleGCG suffix generation and eval sets
data_generation/
  make_safe_responses.py     four refusal styles per behavior (Section 3.1, Figure 2)
  make_dpo_safe_response.py  single safe response per behavior used by the DPO baseline
configs/                 LoRA, trainer, training attacker (GCG 5 tokens, 150 steps), eval attacks, judge
data/                    AdvBench, HarmBench (test + adversarial-training behaviors), generated safe responses
scripts/train/run_paper_config.sh   trains any configuration of Tables 1, 2 and 4
scripts/eval/eval_14cells.sh        PAIR + GCG cfg-1..3 + AmpleGCG cfg-1..3 on AdvBench and HarmBench
analysis/reward_probe.py            Table 3
analysis/plot_reward_separation.py  Figure 3
analysis/plot_refusal_styles.py     Figure 2
```

## Setup

```bash
conda create -n gapdpo python=3.11 && conda activate gapdpo
pip install -r requirements.txt
python download_model.py   
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
```

Models are looked up in `$MODEL_ROOT`, then `./models`, then `~/models`, by directory name
(`Qwen2.5-7B-Instruct`, `HarmBench-Llama-2-13b-cls`, `Mixtral-8x7B-Instruct-v0.1`,
`AmpleGCG-plus-llama2-sourced-vicuna-7b13b-guanaco-7b13b`). The Alpaca utility data is loaded from the
Hugging Face hub (`tatsu-lab/alpaca`).


## Training

```bash
bash scripts/train/run_paper_config.sh <config> [--gpus 0] [--seed 42] [--out-dir results/<config>]
```

## License

Code derived from Adv-ICL is released under its MIT license (`LICENSE`). AdvBench and HarmBench data
keep their own licenses (`data/*/LICENSE`); Alpaca is CC BY-NC 4.0.
