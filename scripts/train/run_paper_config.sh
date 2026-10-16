#!/usr/bin/env bash
# Train one configuration of the paper with the exact arguments of the reported run.
#
#   bash scripts/train/run_paper_config.sh <config> [--gpus 0,1,2,3] [--seed 42] [--out-dir results/<config>]
#
# <config>              paper row                      entry point             GPUs used for the paper run
#   sft                 SFT (Table 1)                  train_sft.py            4 (torchrun), micro 1 x accum 2
#   adv_icl             Adv-ICL [27] (Table 1)         train_adv_icl.py        launcher default 0,1 (device_map=auto)
#   dpo                 DPO (Table 1)                  train_dpo.py            1, micro 1 x accum 8
#   gapdpo_m1k4         GapDPO (M=1,K=4) (Tables 1-4)  train_gapdpo.py         1, micro 4 x accum 2
#   gapdpo_m4k4         GapDPO (M=4,K=4)               train_gapdpo.py         1, micro 4 x accum 2
#   gapdpo_m4k1         GapDPO (M=4,K=1) (Table 2)     train_gapdpo_m4k1.py    1, micro 1 x accum 8
#   gapdpo_wo_vh_m1k4   GapDPO w/o Vh (M=1,K=4)        train_gapdpo.py         8 (torchrun), micro 1 x accum 1
#   gapdpo_wo_vh_m4k4   GapDPO w/o Vh (M=4,K=4)        train_gapdpo.py         8 (torchrun), micro 1 x accum 1
#   gapdpo_m1k4_ah0     alpha_h = 0   (Table 4)        train_gapdpo.py         4 (torchrun), micro 1 x accum 2
#   gapdpo_m1k4_ah0.2   alpha_h = 0.2 (Table 4)        train_gapdpo.py         4 (torchrun), micro 1 x accum 2
#   gapdpo_m1k4_ah0.4   alpha_h = 0.4 (Table 4)        train_gapdpo.py         4 (torchrun), micro 1 x accum 2
#   gapdpo_m1k4_ah0.8   alpha_h = 0.8 (Table 4)        train_gapdpo.py         4 (torchrun), micro 1 x accum 2
#
# Every configuration uses an effective batch of 8 adversarial behaviors and 56 Alpaca examples per
# step; for train_gapdpo.py / train_sft.py the per-rank micro batch is derived from the number of
# GPUs and --gradient-accumulation-steps. --gpus overrides the GPU list (the effective batch is kept).
# The adapter is written to <out-dir>/<config>_fin-model. DRY_RUN=1 only prints the command.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
source "${repo_root}/scripts/_model_paths.sh"

config=${1:-}
[[ -n ${config} ]] || { sed -n 2,26p "$0"; exit 1; }
shift
gpus=""
seed=42
out_dir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) gpus=$2; shift 2 ;;
        --seed) seed=$2; shift 2 ;;
        --out-dir) out_dir=$2; shift 2 ;;
        *) echo "Error: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

dpo_cfg="${repo_root}/configs/train/trainer_dpo_0.3beta.yaml"
gapdpo_args() {  # <chosen indices> <K> <alpha_c> <alpha_h> <accum>
    entry=train_gapdpo.py
    extra=(--dataset harmbench-chosen4 --trainer-cfg-path "${dpo_cfg}"
           --chosen-indices "$1" --num-harmful-samples "$2"
           --chosen-consistency-alpha "$3" --harmful-consistency-alpha "$4"
           --gradient-accumulation-steps "$5")
}

case "${config}" in
    sft)
        entry=train_sft.py; default_gpus=0,1,2,3
        extra=(--dataset harmbench-chosen4 --trainer-cfg-path "${dpo_cfg}" --chosen-index 2 --gradient-accumulation-steps 2) ;;
    adv_icl)
        entry=train_adv_icl.py; default_gpus=0,1
        extra=(--dataset harmbench --trainer-cfg-path "${repo_root}/configs/train/trainer.yaml") ;;
    dpo)
        entry=train_dpo.py; default_gpus=0
        extra=(--dataset harmbench-chosen --trainer-cfg-path "${dpo_cfg}" --gradient-accumulation-steps 8) ;;
    gapdpo_m4k1)
        entry=train_gapdpo_m4k1.py; default_gpus=0
        extra=(--dataset harmbench-chosen4 --trainer-cfg-path "${dpo_cfg}" --gradient-accumulation-steps 8) ;;
    gapdpo_m1k4)        default_gpus=0;               gapdpo_args 2       4 0.1 0.1 2 ;;
    gapdpo_m4k4)        default_gpus=0;               gapdpo_args 0,1,2,3 4 0.1 0.1 2 ;;
    gapdpo_wo_vh_m1k4)  default_gpus=0,1,2,3,4,5,6,7; gapdpo_args 2       4 0.1 0.0 1 ;;
    gapdpo_wo_vh_m4k4)  default_gpus=0,1,2,3,4,5,6,7; gapdpo_args 0,1,2,3 4 0.1 0.0 1 ;;
    gapdpo_m1k4_ah0|gapdpo_m1k4_ah0.2|gapdpo_m1k4_ah0.4|gapdpo_m1k4_ah0.8)
        default_gpus=0,1,2,3; gapdpo_args 2 4 0.0 "${config##*_ah}" 2 ;;
    *) echo "Error: unknown config '${config}'" >&2; sed -n 5,18p "$0" >&2; exit 2 ;;
esac

gpus=${gpus:-${default_gpus}}
out_dir=${out_dir:-${repo_root}/results/${config}}
[[ ${out_dir} == /* ]] || out_dir="${repo_root}/${out_dir}"
model_id=$(resolve_model_dir "Qwen2.5-7B-Instruct")
if [[ ${DRY_RUN:-0} != 1 ]]; then
    [[ -d ${model_id} ]] || { echo "Error: missing ${model_id} (see README, Setup)" >&2; exit 1; }
    mkdir -p "${out_dir}"
fi

export CUDA_VISIBLE_DEVICES=${gpus}
IFS=',' read -ra gpu_array <<< "${gpus}"
case "${entry}" in
    train_sft.py|train_gapdpo.py) launcher=(torchrun --standalone --nproc_per_node "${#gpu_array[@]}") ;;
    *) launcher=(python) ;;
esac

cd "${repo_root}/src"
cmd=("${launcher[@]}" "${entry}"
    --model-id "${model_id}"
    --lora-cfg-path "${repo_root}/configs/train/lora.yaml"
    --datacollator qwen2-chat
    --utilityset alpaca
    --atker-cfg-path "${repo_root}/configs/train/atker/cfg-1.yaml"
    --save-dir "${out_dir}"
    --save-name "${config}"
    --random-seed "${seed}"
    "${extra[@]}")
echo "[$(date '+%F %T')] CUDA_VISIBLE_DEVICES=${gpus}"
echo "${cmd[*]}"
[[ ${DRY_RUN:-0} == 1 ]] && exit 0
"${cmd[@]}"
