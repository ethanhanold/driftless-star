"""Select existing Stage 1 outputs or produce them in a solver container."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = REPO_ROOT / "inputs" / "stage1-tests"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register Stage 1 artifact selection and container options."""
    group = parser.getgroup("stage1", "standalone Stage 1 physics tests")
    group.addoption("--stage1-run", choices=["vmec-jax"], help="Run this solver in its production container.")
    group.addoption("--stage1-wout", action="append", default=[], help="Existing wout path; may be repeated.")
    group.addoption("--stage1-case", action="append", default=[], help="Case name after input; default: all cases.")
    group.addoption("--stage1-runtime", choices=["docker", "apptainer"], default="docker")
    group.addoption("--stage1-device", choices=["cpu", "gpu"], default="cpu")
    group.addoption("--stage1-image", help="Docker image, native Apptainer SIF, or ORAS URI override.")
    group.addoption("--stage1-output-dir", default="outputs/stage1-tests", help="Directory for fresh runs and logs.")


def pytest_configure(config: pytest.Config) -> None:
    """Reject conflicting modes and invalid explicit selections before running tests."""
    run = config.getoption("stage1_run")
    paths = config.getoption("stage1_wout")
    cases = config.getoption("stage1_case")
    if run and paths:
        raise pytest.UsageError("--stage1-run and --stage1-wout are mutually exclusive")
    if cases and not run:
        raise pytest.UsageError("--stage1-case requires --stage1-run")
    for case in cases:
        if case not in {path.name.removeprefix("input.") for path in CASE_DIR.glob("input.*")}:
            raise pytest.UsageError(f"Unknown Stage 1 case: {case}")
    for path in paths:
        if not Path(path).expanduser().is_file():
            raise pytest.UsageError(f"Stage 1 wout does not exist: {path}")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Share one artifact per selected case across all physics assertions."""
    if "wout_path" not in metafunc.fixturenames:
        return
    config = metafunc.config
    if config.getoption("stage1_run"):
        cases = config.getoption("stage1_case") or sorted(
            path.name.removeprefix("input.") for path in CASE_DIR.glob("input.*")
        )
        if not cases:
            raise pytest.UsageError(f"No Stage 1 input cases found in {CASE_DIR}")
        selections = list(dict.fromkeys(cases))
    else:
        selections = config.getoption("stage1_wout") or [None]
    metafunc.parametrize("wout_path", selections, indirect=True, scope="session")


@pytest.fixture(scope="session")
def wout_path(request: pytest.FixtureRequest) -> Path:
    """Return an explicit existing artifact or run a case once in a fresh directory."""
    if request.param is None:
        pytest.skip("Select --stage1-wout or --stage1-run to test an equilibrium")
    config = request.config
    solver = config.getoption("stage1_run")
    if not solver:
        return Path(request.param).expanduser().resolve()

    case = request.param
    case_dir = Path(config.getoption("stage1_output_dir")).expanduser().resolve() / solver / case
    case_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=case_dir))
    output = run_dir / "wout.nc"
    runtime = config.getoption("stage1_runtime")
    device = config.getoption("stage1_device")
    image = config.getoption("stage1_image")
    if runtime == "docker":
        image = image or f"ghcr.io/driftless-star/driftless-star:stage-1-vmec-{device}"
        command = [
            "docker", "run", "--rm", "--pull=missing", "--user", f"{os.getuid()}:{os.getgid()}",
            "--env", "HOME=/stage1-output", "--volume", f"{REPO_ROOT}:/work:ro",
            "--volume", f"{run_dir}:/stage1-output", "--workdir", "/work",
        ]
        if device == "gpu":
            command += ["--gpus", "all"]
    else:
        image = image or f"oras://ghcr.io/driftless-star/driftless-star:apptainer-stage-1-vmec-{device}"
        if "://" not in image:
            image = str(Path(image).expanduser().resolve())
        command = [
            "apptainer", "exec", "--unsquash", "--bind", f"{REPO_ROOT}:/work:ro",
            "--bind", f"{run_dir}:/stage1-output", "--pwd", "/work", "--env", "HOME=/stage1-output",
        ]
        if device == "gpu":
            command.append("--nv")
    command += [
        image, "python", "stages/stage1-equilibrium/run_vmec_jax.py",
        "--input", f"/work/inputs/stage1-tests/input.{case}", "--output", "/stage1-output/wout.nc",
    ]
    (run_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    log_path = run_dir / "solver.log"
    with log_path.open("w") as log:
        try:
            subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            pytest.fail(f"Stage 1 {solver}/{case} failed: {exc}; see {log_path}", pytrace=False)
    if not output.is_file():
        pytest.fail(f"Stage 1 did not produce {output}; see {log_path}", pytrace=False)
    return output
