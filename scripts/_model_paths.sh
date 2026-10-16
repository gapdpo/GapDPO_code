set -euo pipefail

_model_paths_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ADV_ICL_REPO_ROOT=$(cd "${_model_paths_dir}/.." && pwd)

candidate_model_roots=()
if [[ -n ${MODEL_ROOT:-} ]]; then
    candidate_model_roots+=("$(cd "${MODEL_ROOT}" 2>/dev/null && pwd || printf '%s' "${MODEL_ROOT}")")
fi
candidate_model_roots+=("${ADV_ICL_REPO_ROOT}/models")
candidate_model_roots+=("${HOME}/models")

resolve_model_dir() {
    local model_name=$1
    local root
    for root in "${candidate_model_roots[@]}"; do
        if [[ -d "${root}/${model_name}" ]]; then
            cd "${root}/${model_name}" && pwd
            return 0
        fi
    done

    printf '%s\n' "${candidate_model_roots[0]}/${model_name}"
}
