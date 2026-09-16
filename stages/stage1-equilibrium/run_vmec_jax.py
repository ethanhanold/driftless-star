"""Run vmec_jax with its CLI defaults and export wout with explicit solve status."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil


def main(argv: list[str] | None = None) -> int:
    """Solve one input and write its native wout plus convergence metadata.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; defaults to the process arguments.

    Returns
    -------
    int
        Zero for convergence, one for an unconverged diagnostic output.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"Input does not exist: {input_path}")
    for destination in (output_path, output_path.with_suffix(".input"), output_path.with_suffix(".json")):
        if destination.exists():
            parser.error(f"Refusing to overwrite existing run artifact: {destination}")

    import numpy as np
    from netCDF4 import Dataset
    from vmec_jax.driver import (
        default_non_autodiff_solver_policy,
        run_fixed_boundary,
        write_wout_from_fixed_boundary_run,
    )
    from vmec_jax.namelist import read_indata

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, output_path.with_suffix(".input"))
    indata = read_indata(input_path)
    solver_mode, performance_mode = default_non_autodiff_solver_policy(indata)
    # Match vmec_jax 0.0.4's CLI; omitting max_iter preserves the input's budgets.
    settings = dict(
        step_size=None,
        jit_forces="auto",
        solver_mode=solver_mode,
        performance_mode=performance_mode,
        cli_fixed_boundary_mode=True,
    )
    metadata = {"input": str(input_path), "vmec_jax_version": version("vmec-jax"), "settings": settings}
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    run = run_fixed_boundary(str(input_path), **settings)
    converged = run.result.diagnostics.get("converged")
    if not isinstance(converged, bool):
        raise RuntimeError("vmec_jax did not report a boolean convergence result")
    write_wout_from_fixed_boundary_run(output_path, run, include_fsq=True)

    # The pinned 0.0.4 writer omits these fields specified by the Stage 1 contract.
    with Dataset(output_path, "a") as dataset:
        dataset.createVariable("ier_flag", "i4").assignValue(0 if converged else 1)
        dataset.createVariable("vmec_jax_converged__logical__", "i4").assignValue(int(converged))
        status = "converged" if converged else "not_converged"
        dataset.createDimension("stage1_status_length", len(status))
        variable = dataset.createVariable("vmec_jax_status", "S1", ("stage1_status_length",))
        variable[:] = np.frombuffer(status.encode("ascii"), dtype="S1")
    print(f"{status}: {output_path}")
    return 0 if converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
