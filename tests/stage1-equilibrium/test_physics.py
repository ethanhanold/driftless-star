"""Shared Stage 1 acceptance checks on actual equilibrium outputs."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import warnings

import numpy as np
import pytest
import xarray as xr

'''Tolerances for iota checks. These are not physical tolerances, but rather
numerical tolerances for the sampled iota profile. They are chosen to be
reasonably strict, but not so strict as to fail valid equilibria due to numerical noise or solver artifacts.'''
IOTA_ZERO_TOL = 1.0e-10
IOTA_TV_RATIO_TOL = 3.0
IOTA_STEP_ATOL = 1.0e-12
IOTA_STEP_RTOL = 1.0e-8
IOTA_MAX_CONSECUTIVE_SAWTOOTH_FLIPS = 3
IOTA_CHEB_DEGREE = 15
IOTA_SPECTRAL_DECAY_TOL = 0.0
IOTA_MAX_RATIONAL_DENOMINATOR = 6
IOTA_NEAR_RATIONAL_TOL = 1.0e-3




def get_max_consecutive_sawtooth_flips(profile: np.ndarray, atol: float, rtol: float) -> int:
    """Count consecutive reversals between significant adjacent differences.

    Parameters
    ----------
    profile : np.ndarray
        Finite, one-dimensional sampled profile.
    atol, rtol : float
        Nonnegative step thresholds; a difference is significant when its
        magnitude exceeds ``atol + rtol * max(abs(profile))``.

    Returns
    -------
    int
        Longest run of reversals. Flat or negligible steps break the run.
    """
    if profile.size < 3:
        return 0
    delta = np.diff(profile)
    threshold = atol + rtol * np.max(np.abs(profile))
    directions = np.where(np.abs(delta) > threshold, np.sign(delta), 0)

    flips = (directions[:-1] * directions[1:]) < 0
    max_consecutive = 0
    current_consecutive = 0

    for flip in flips:
        if flip:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    return max_consecutive


def get_spectral_decay(s: np.ndarray, profile: np.ndarray, degree: int) -> float | None:
    """Estimate decay of Chebyshev coefficients of a radial profile."""
    x = 2 * (s - s[0]) / (s[-1] - s[0]) - 1
    centered = profile - profile[0]
    
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.RankWarning)
        try:
            coefficients = np.polynomial.chebyshev.chebfit(x, centered, degree)
        except np.exceptions.RankWarning:
            return None

    amplitudes = np.abs(coefficients[1:])
    decay_rate, _ = np.polyfit(np.arange(0, degree), np.log(amplitudes), 1)
    return float(decay_rate)


@pytest.fixture(scope="session")
def wout(wout_path: Path) -> Iterator[xr.Dataset]:
    with xr.open_dataset(wout_path, engine="netcdf4") as dataset:
        yield dataset


def _normalized_toroidal_flux(wout: xr.Dataset) -> np.ndarray:
    """Validate and return normalized toroidal flux s."""
    for name in ("ns", "phi"):
        assert name in wout, f"Missing required wout field: {name}"
    assert wout.ns.ndim == 0 and wout.ns.dtype.kind in "iu", "ns must be an integer scalar"
    ns = int(wout.ns.item())
    assert ns >= 3, "ns must be at least 3"
    assert wout["phi"].dims == ("radius",), "phi must use the full-mesh radius dimension"
    assert wout["phi"].shape == (ns,), f"phi must have shape ({ns},)"
    assert wout["phi"].dtype.kind == "f", "phi must contain floating-point values"
    phi = wout.phi.values
    assert np.isfinite(phi).all(), "phi must be finite"
    assert phi[0] == 0 and phi[-1] != 0, "phi must start at zero and have nonzero edge flux"
    s = phi / phi[-1]
    assert np.isfinite(s).all() and (np.diff(s) > 0).all(), "normalized toroidal flux must increase from axis to edge"

    return s


def _iota(wout: xr.Dataset) -> np.ndarray:
    """Validate and return signed full-mesh iota."""
    s = _normalized_toroidal_flux(wout)
    assert "iotaf" in wout, f"Missing required wout field: iotaf"
    assert wout["iotaf"].dims == ("radius",), "iotaf must use the full-mesh radius dimension"
    assert wout["iotaf"].shape == (len(s),), f"iotaf must have shape ({len(s)},)"
    assert wout["iotaf"].dtype.kind == "f", "iotaf must contain floating-point values"
    iota = wout.iotaf.values

    bad = np.flatnonzero(~np.isfinite(iota))
    assert bad.size == 0, f"Nonfinite iota at indices {bad}, s={s[bad]}, iota={iota[bad]}"

    return iota    


def test_iota_nonzero(wout: xr.Dataset) -> None:
    """Require every sampled magnitude to exceed the explicit zero threshold."""
    iota = _iota(wout)
    s = _normalized_toroidal_flux(wout)
    bad = np.flatnonzero(~(np.abs(iota) > IOTA_ZERO_TOL))
    assert bad.size == 0, f"|iota| must not be zero; exceeded tolerance {IOTA_ZERO_TOL}: indices {bad}, s={s[bad]}, iota={iota[bad]}"


def test_iota_consistent_sign(wout: xr.Dataset) -> None:
    """Requie iota to have consistent sign across the radial grid."""
    iota = _iota(wout)
    assert (iota > 0).all() or (iota < 0).all(), f"Iota must have one nonzero sign"


def test_iota_variation(wout: xr.Dataset) -> None:
    """Bound repeated oscillations relative to the full profile range."""
    iota = _iota(wout)
    
    iota_TV = np.sum(np.abs(np.diff(iota)))
    if iota_TV <= 1e-3:
        return
    ratio_TV = iota_TV / np.ptp(iota)
    assert ratio_TV < IOTA_TV_RATIO_TOL, (
        f"Excessive iota oscillations: TV={iota_TV}, TV/range={ratio_TV} >= {IOTA_TV_RATIO_TOL}"
    )


def test_iota_sawtoothing(wout: xr.Dataset) -> None:
    """Verifies that the iota profile does not exhibit rapid grid-scale sawtoothing."""
    iota = _iota(wout)
    max_consecutive = get_max_consecutive_sawtooth_flips(iota, atol=IOTA_STEP_ATOL, rtol=IOTA_STEP_RTOL)
    assert max_consecutive <= IOTA_MAX_CONSECUTIVE_SAWTOOTH_FLIPS, (
        f"Too many consecutive sawtooth flips in iota: {max_consecutive} > {IOTA_MAX_CONSECUTIVE_SAWTOOTH_FLIPS}"
    )


def test_iota_spectral_decay(wout: xr.Dataset) -> None:
    """Require Chebyshev coefficient magnitudes to decrease on average with degree."""
    s = _normalized_toroidal_flux(wout)
    iota = _iota(wout)
    decay_rate = get_spectral_decay(s, iota, degree=IOTA_CHEB_DEGREE)
    assert decay_rate < IOTA_SPECTRAL_DECAY_TOL, (
        "Expected spectral decay of smooth iota profile. "
        f"Iota spectral decay rate {decay_rate} is not less than tolerance {IOTA_SPECTRAL_DECAY_TOL}"
    )


def test_iota_nearly_rational(wout: xr.Dataset) -> None:
    """Require iota to avoid low-order rationals.

    When iota = m/n, a field line closes after m poloidal and n toroidal turns.
    Small magnetic-field perturbations can then reinforce each other on
    repeated turns and form magnetic islands. These islands can increase
    radial heat and particle transport and reduce plasma confinement.
    Avoiding low-order rationals reduces exposure to these resonances.
    """
    iota = _iota(wout)
    s = _normalized_toroidal_flux(wout)
    denominators = np.arange(1, IOTA_MAX_RATIONAL_DENOMINATOR + 1)

    numerators = np.rint(iota[:, None] * denominators)
    distances = np.abs(iota[:, None] - numerators / denominators)
    radial_idx, rational_idx = np.unravel_index(np.argmin(distances), distances.shape)
    distance = distances[radial_idx, rational_idx]
    numerator = int(numerators[radial_idx, rational_idx])
    denominator = int(denominators[rational_idx])
    assert distance > IOTA_NEAR_RATIONAL_TOL, (
        f"Iota at index {radial_idx}, s={s[radial_idx]}, iota={iota[radial_idx]} is within "
        f"{distance} of {numerator}/{denominator}; require distance > {IOTA_NEAR_RATIONAL_TOL}"
    )

    lower = np.minimum(iota[:-1], iota[1:])
    upper = np.maximum(iota[:-1], iota[1:])
    for denominator in denominators:
        numerators = np.ceil(lower * denominator)
        crossed = np.flatnonzero(numerators / denominator <= upper)
        assert crossed.size == 0, f"Iota crosses the rational {int(numerators[crossed[0]])}/{denominator}"
