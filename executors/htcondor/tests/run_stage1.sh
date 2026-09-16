#!/usr/bin/env bash
# Batch executable for stage1.sub; HTCondor invokes this with the selected test arguments.
# Arguments: all|CASE[,CASE...] [cpu|gpu] [SIF_OR_ORAS_URI]
# The submit file transfers the repository-relative files into the job scratch directory.
# Pixi and its locked test environment are installed on the worker; Apptainer must
# already be available there. The default solver image is downloaded into job scratch.

set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $0 all|CASE[,CASE...] [cpu|gpu] [SIF_OR_ORAS_URI]"
    exit 0
fi
if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 all|CASE[,CASE...] [cpu|gpu] [SIF_OR_ORAS_URI]" >&2
    exit 2
fi

: "${_CONDOR_SCRATCH_DIR:?Run this script inside an allocated HTCondor job.}"
cd "${_CONDOR_SCRATCH_DIR}"
job_root="$(pwd -P)"
mkdir -p stage1-results
result_dir="$(mktemp -d "${job_root}/stage1-results/run-XXXXXX")"

# Archive on both setup and test failures. Only results are transferred back,
# excluding installed environments, downloaded containers, and caches.
save_results() {
    local status=$?
    trap - EXIT
    printf '%s\n' "${status}" > "${result_dir}/runner-exit-code.txt"
    if ! tar -czf "${job_root}/stage1-results.tar.gz" -C "${job_root}" stage1-results; then
        echo "Failed to archive Stage 1 results in ${job_root}." >&2
        if [[ ${status} -eq 0 ]]; then
            status=1
        fi
    fi
    echo "Stage 1 runner exit status: ${status}; results: ${result_dir}"
    exit "${status}"
}
trap save_results EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

test_case="$1"
device="${2:-cpu}"
if [[ "${device}" != "cpu" && "${device}" != "gpu" ]]; then
    echo "Device must be cpu or gpu; got: ${device}" >&2
    exit 2
fi
case_args=()
if [[ "${test_case}" != "all" ]]; then
    if [[ ! "${test_case}" =~ ^[a-zA-Z0-9_.-]+(,[a-zA-Z0-9_.-]+)*$ ]]; then
        echo "Cases must be 'all' or comma-separated case names without spaces." >&2
        exit 2
    fi
    IFS=',' read -r -a selected_cases <<< "${test_case}"
    for selected_case in "${selected_cases[@]}"; do
        if [[ ! -f "inputs/stage1-tests/input.${selected_case}" ]]; then
            echo "Missing transferred input: inputs/stage1-tests/input.${selected_case}" >&2
            exit 2
        fi
        case_args+=(--stage1-case "${selected_case}")
    done
fi
solver_image="${3:-oras://ghcr.io/driftless-star/driftless-star:apptainer-stage-1-vmec-${device}}"
if ! command -v apptainer >/dev/null 2>&1; then
    echo "Apptainer must be installed on the execute node." >&2
    exit 127
fi

export TMPDIR="${job_root}/tmp"
export XDG_CACHE_HOME="${job_root}/cache"
export PIXI_HOME="${job_root}/pixi"
export PIXI_CACHE_DIR="${XDG_CACHE_HOME}/pixi"
export APPTAINER_CACHEDIR="${job_root}/apptainer-cache"
export APPTAINER_TMPDIR="${TMPDIR}"
mkdir -p "${TMPDIR}" "${PIXI_CACHE_DIR}" "${APPTAINER_CACHEDIR}"

if ! command -v pixi >/dev/null 2>&1; then
    curl -fsSL https://pixi.sh/install.sh -o "${TMPDIR}/install-pixi.sh"
    PIXI_NO_PATH_UPDATE=1 TMP_DIR="${TMPDIR}" bash "${TMPDIR}/install-pixi.sh" \
        2>&1 | tee "${result_dir}/pixi-bootstrap.log"
    export PATH="${PIXI_HOME}/bin:${PATH}"
fi

echo "Installing the Stage 1 test environment; log: ${result_dir}/environment.log"
pixi install --manifest-path stages/pixi.toml -e stage-1-tests --locked \
    2>&1 | tee "${result_dir}/environment.log"

echo "Running vmec-jax / ${test_case} on ${device}; solver logs will be under ${result_dir}/vmec-jax/"
set +e
pixi run --manifest-path stages/pixi.toml -e stage-1-tests --locked \
    python -u -m pytest tests/stage1-equilibrium/test_physics.py \
    --stage1-run vmec-jax "${case_args[@]}" \
    --stage1-runtime apptainer --stage1-device "${device}" \
    --stage1-image "${solver_image}" --stage1-output-dir "${result_dir}" \
    -s -v 2>&1 | tee "${result_dir}/pytest.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${pipeline_status[0]}" > "${result_dir}/pytest-exit-code.txt"
if [[ ${pipeline_status[0]} -ne 0 ]]; then
    exit "${pipeline_status[0]}"
fi
exit "${pipeline_status[1]}"
