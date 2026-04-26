"""eos objective. No LAMMPS — feed synthetic (P, V) directly via SimResult.extras."""
import pytest

from pyfield.config.schema import TargetCfg
from pyfield.objectives import build_objective
from pyfield.objectives.base import ObjectiveContext
from pyfield.simulations.base import SimResult


_ATM_TO_GPA = 1.01325e-4


def _npt_result(p_atm: float, vol_a3: float) -> SimResult:
    return SimResult(
        sim_id="x", energy=0.0,
        extras={
            "dump_file": "/dev/null",
            "pressure": float(p_atm),
            "mean_volume": float(vol_a3),
        },
    )


def test_eos_known_bulk_modulus():
    """Linear V vs P. Choose dV/dP so B has a known value.

    B [GPa] = -V0 / (dV/dP) * (atm→GPa)
    Pick V0 = 1000, slope = -0.05 Å³/atm  →  B = 1000 / 0.05 * 1.01325e-4 = 2.0265 GPa.
    """
    sims = {
        "p0":  _npt_result(p_atm=0,    vol_a3=1000.0),
        "p10": _npt_result(p_atm=10,   vol_a3=999.5),
        "p20": _npt_result(p_atm=20,   vol_a3=999.0),
    }
    expected_B = 1000 / 0.05 * _ATM_TO_GPA
    tgt = TargetCfg.model_validate({
        "kind": "eos", "weight": 2.0,
        "simulations": list(sims.keys()),
        "target": expected_B,
    })
    obj = build_objective(tgt)
    assert obj.compute(ObjectiveContext(sim_results=sims)) == pytest.approx(expected_B, rel=1e-6)
    assert obj.residual(ObjectiveContext(sim_results=sims)) == pytest.approx(0.0, abs=1e-12)


def test_eos_residual_grows_with_target_distance():
    sims = {
        "p0":  _npt_result(0, 1000.0),
        "p10": _npt_result(10, 999.5),
    }
    expected_B = 1000 / 0.05 * _ATM_TO_GPA
    tgt = TargetCfg.model_validate({
        "kind": "eos", "weight": 1.0,
        "simulations": list(sims.keys()),
        "target": expected_B + 1.0,   # off by 1 GPa
    })
    obj = build_objective(tgt)
    assert obj.residual(ObjectiveContext(sim_results=sims)) == pytest.approx(1.0, rel=1e-6)


def test_eos_requires_two_simulations():
    with pytest.raises(ValueError, match="at least 2"):
        TargetCfg.model_validate({
            "kind": "eos", "weight": 1.0,
            "simulations": ["only_one"], "target": 1.0,
        })
        # the validation happens in EosObjective.__init__; trigger build:
        from pyfield.objectives import build_objective
        from pyfield.config.schema import TargetCfg as T
        build_objective(T.model_validate({
            "kind": "eos", "weight": 1.0,
            "simulations": ["only_one"], "target": 1.0,
        }))


def test_eos_missing_pressure_or_volume_rejected():
    sims = {
        "p0":  _npt_result(0, 1000.0),
        "p1":  SimResult(sim_id="p1", energy=0.0, extras={}),  # missing
    }
    tgt = TargetCfg.model_validate({
        "kind": "eos", "weight": 1.0,
        "simulations": ["p0", "p1"], "target": 1.0,
    })
    obj = build_objective(tgt)
    with pytest.raises(RuntimeError, match="mean_volume"):
        obj.compute(ObjectiveContext(sim_results=sims))
