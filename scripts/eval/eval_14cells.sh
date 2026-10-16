#!/usr/bin/env bash
set -euo pipefail

# Evaluate one model on the 14 attack cells of Tables 1, 2 and 4:
#   PAIR + GCG cfg-1..3 + AmpleGCG cfg-1..3, each on AdvBench-first50 and HarmBench-test50.
# Outputs go to <out-dir>/eval-pair, <out-dir>/eval-gcg/{1,2,3}, <out-dir>/eval-amplegcg/{1,2,3}.
# Each judge log ends with "attack success rate: X%" (a behavior counts as attacked when any of its
# 10 sampled responses is judged harmful). Finished evalsets and judgements are skipped on re-run.
#
#   bash scripts/eval/eval_14cells.sh --gpu 0 --out-dir results/gapdpo_m1k4 \
#       --lora-cfg results/gapdpo_m1k4/gapdpo_m1k4_lora-cfg.yaml \
#       --adapter results/gapdpo_m1k4/gapdpo_m1k4_fin-model/adapter_model.safetensors \
#       [--cells pair,gcg,amplegcg] [--datasets advbench-first50,harmbench-test50] [--seed 43]
#
# Omit --lora-cfg/--adapter to evaluate the initial model.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
source "${repo_root}/scripts/_model_paths.sh"

src_path="${repo_root}/src"
python_bin="${PYTHON_BIN:-python}"
gpu=""
out_dir=""
lora_cfg=""
adapter=""
cells="pair,gcg,amplegcg"
datasets="advbench-first50,harmbench-test50"
seed=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) gpu=$2; shift 2 ;;
        --out-dir) out_dir=$2; shift 2 ;;
        --lora-cfg) lora_cfg=$2; shift 2 ;;
        --adapter) adapter=$2; shift 2 ;;
        --cells) cells=$2; shift 2 ;;
        --datasets) datasets=$2; shift 2 ;;
        --seed) seed=$2; shift 2 ;;
        *) echo "Error: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

abspath() {
    if [[ -z $1 || $1 == /* ]]; then printf '%s' "$1"; else printf '%s/%s' "${repo_root}" "$1"; fi
}
out_dir=$(abspath "${out_dir}")
lora_cfg=$(abspath "${lora_cfg}")
adapter=$(abspath "${adapter}")

if [[ ! ${gpu} =~ ^[0-9]+$ || -z ${out_dir} ]]; then
    echo "Error: --gpu and --out-dir are required" >&2
    exit 2
fi
if [[ -n ${adapter} || -n ${lora_cfg} ]]; then
    for path in "${adapter}" "${lora_cfg}"; do
        if [[ ! -s ${path} ]]; then
            echo "Error: missing ${path}" >&2
            exit 1
        fi
    done
fi

model_id=$(resolve_model_dir "Qwen2.5-7B-Instruct")
judge_cfg_path="${repo_root}/configs/eval/judge-bs5.yaml"
[[ -f ${judge_cfg_path} ]] || judge_cfg_path="${repo_root}/configs/eval/judge.yaml"

model_args=(--model-id "${model_id}" --datacollator qwen2-chat)
if [[ -n ${adapter} ]]; then
    model_args+=(--lora-cfg-path "${lora_cfg}" --model-resume-path "${adapter}")
fi
seed_args=()
if [[ -n ${seed} ]]; then
    seed_args=(--random-seed "${seed}")
fi

export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ds_name_for() {
    case "$1" in
        advbench-first50) printf 'advbench' ;;
        harmbench-test50) printf 'harmbench' ;;
    esac
}

judge_succeeded() {
    [[ -f $1 ]] && grep -q "attack success rate:" "$1"
}

run_judge() {
    local save_path=$1 ds_name=$2
    judge_succeeded "${save_path}/judge-${ds_name}_log.txt" && return 0
    (cd "${src_path}" && "${python_bin}" evaluate.py \
        --judger-cfg-path "${judge_cfg_path}" \
        --evalset-path "${save_path}/build-${ds_name}_evalset.json" \
        --save-dir "${save_path}" \
        --exp-type judge-evalset \
        --save-name "judge-${ds_name}")
}

run_build() {
    local save_path=$1 dataset=$2 atker_cfg=$3 ds_name
    ds_name=$(ds_name_for "${dataset}")
    mkdir -p "${save_path}"
    if judge_succeeded "${save_path}/judge-${ds_name}_log.txt"; then
        echo "[GPU ${gpu}] skip completed ${save_path} ${ds_name}"
        return 0
    fi
    if [[ ! -s "${save_path}/build-${ds_name}_evalset.json" ]]; then
        echo "[$(date '+%F %T')] [GPU ${gpu}] build ${save_path} ${ds_name}"
        (cd "${src_path}" && "${python_bin}" evaluate.py \
            "${model_args[@]}" "${seed_args[@]}" \
            --dataset "${dataset}" \
            --evalset-cfg-path "${repo_root}/configs/eval/evalset.yaml" \
            --atker-cfg-path "${atker_cfg}" \
            --save-dir "${save_path}" \
            --exp-type build-evalset \
            --save-name "build-${ds_name}")
    fi
    run_judge "${save_path}" "${ds_name}"
}

run_amplegcg() {
    local save_path=$1 dataset=$2 cfg=$3 ds_name sfx_len
    ds_name=$(ds_name_for "${dataset}")
    case "${cfg}" in 1) sfx_len=5 ;; 2) sfx_len=10 ;; 3) sfx_len=20 ;; esac
    mkdir -p "${save_path}"
    if judge_succeeded "${save_path}/judge-${ds_name}_log.txt"; then
        echo "[GPU ${gpu}] skip completed ${save_path} ${ds_name}"
        return 0
    fi
    if [[ ! -s "${save_path}/pre-evalset-${ds_name}-${cfg}.json" ]]; then
        echo "[$(date '+%F %T')] [GPU ${gpu}] AmpleGCG suffixes ${save_path} ${ds_name}"
        (cd "${src_path}" && "${python_bin}" evaluate_amplegcg.py \
            "${seed_args[@]}" \
            --dataset "${dataset}" \
            --exp-type build-pre-evalset \
            --amplegcg-sfx-len "${sfx_len}" \
            --save-dir "${save_path}" \
            --save-name "pre-evalset-${ds_name}-${cfg}")
    fi
    if [[ ! -s "${save_path}/build-${ds_name}_evalset.json" ]]; then
        echo "[$(date '+%F %T')] [GPU ${gpu}] build ${save_path} ${ds_name}"
        (cd "${src_path}" && "${python_bin}" evaluate_amplegcg.py \
            "${model_args[@]}" "${seed_args[@]}" \
            --pre-evalset-path "${save_path}/pre-evalset-${ds_name}-${cfg}.json" \
            --evalset-cfg-path "${repo_root}/configs/eval/evalset.yaml" \
            --save-dir "${save_path}" \
            --exp-type build-evalset \
            --save-name "build-${ds_name}")
    fi
    run_judge "${save_path}" "${ds_name}"
}

IFS=',' read -ra cell_array <<< "${cells}"
IFS=',' read -ra dataset_array <<< "${datasets}"
for cell in "${cell_array[@]}"; do
    for dataset in "${dataset_array[@]}"; do
        case "${cell}" in
            pair)
                run_build "${out_dir}/eval-pair" "${dataset}" "${repo_root}/configs/eval/pair/cfg-5.yaml" ;;
            gcg)
                for cfg in 1 2 3; do
                    run_build "${out_dir}/eval-gcg/${cfg}" "${dataset}" "${repo_root}/configs/eval/gcg/cfg-${cfg}.yaml"
                done ;;
            amplegcg)
                for cfg in 1 2 3; do
                    run_amplegcg "${out_dir}/eval-amplegcg/${cfg}" "${dataset}" "${cfg}"
                done ;;
            *) echo "Error: unknown cell '${cell}'" >&2; exit 2 ;;
        esac
    done
done
echo "[$(date '+%F %T')] [GPU ${gpu}] done: ${out_dir} (${cells})"
