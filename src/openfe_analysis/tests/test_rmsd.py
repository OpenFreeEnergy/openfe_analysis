import netCDF4 as nc
from openfe_analysis.rmsd import gather_rms_data
from numpy.testing import assert_allclose
import numpy as np
import pytest


@pytest.mark.flaky(reruns=3)
def test_gather_rms_data_regression(simulation_nc, hybrid_system_pdb):
    output = gather_rms_data(
        hybrid_system_pdb,
        simulation_nc,
        skip=100,
    )

    assert_allclose(output["time(ps)"], [0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    assert len(output["protein_RMSD"]) == 11
    assert_allclose(
        output["protein_RMSD"][0],
        [0.0, 1.088, 1.009, 1.120, 1.026, 1.167],
        rtol=1e-3,
    )
    assert len(output["ligand_RMSD"]) == 11
    assert_allclose(
        output["ligand_RMSD"][0],
        [0.0, 0.9434, 0.8068, 0.8255, 1.2313, 0.7186],
        rtol=1e-3,
    )
    assert len(output["ligand_wander"]) == 11
    assert_allclose(
        output["ligand_wander"][0],
        [0.0, 0.8128, 0.5010, 0.6392, 1.1071, 0.3021],
        rtol=1e-3,
    )
    assert len(output["protein_2D_RMSD"]) == 11
    # 15 entries because 6 * 6 frames // 2
    assert len(output["protein_2D_RMSD"][0]) == 15
    assert_allclose(
        output["protein_2D_RMSD"][0][:6],
        [1.0884, 1.0099, 1.1200, 1.0267, 1.1673, 1.2378],
        rtol=1e-3,
    )


@pytest.mark.flaky(reruns=3)
def test_gather_rms_data_regression_skippednc(simulation_skipped_nc, hybrid_system_skipped_pdb):
    output = gather_rms_data(
        hybrid_system_skipped_pdb,
        simulation_skipped_nc,
        skip=None,
    )

    assert_allclose(output["time(ps)"], [0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    assert len(output["protein_RMSD"]) == 11
    assert_allclose(
        output["protein_RMSD"][0],
        [0, 1.176307, 1.203364, 1.486987, 1.17462, 1.143457],
        rtol=1e-3,
    )
    assert len(output["ligand_RMSD"]) == 11
    assert_allclose(
        output["ligand_RMSD"][0],
        [0.0, 1.066418, 1.314562, 1.051574, 0.451605, 0.706698],
        rtol=1e-3,
    )
    assert len(output["ligand_wander"]) == 11
    assert_allclose(
        output["ligand_wander"][0],
        [0.0, 0.726258, 0.628337, 0.707796, 0.329651, 0.483037],
        rtol=1e-3,
    )
    assert len(output["protein_2D_RMSD"]) == 11
    # 15 entries because 6 * 6 frames // 2
    assert len(output["protein_2D_RMSD"][0]) == 15
    assert_allclose(
        output["protein_2D_RMSD"][0][:6],
        [1.176307, 1.203364, 1.486987, 1.17462, 1.143457, 1.244173],
        rtol=1e-3,
    )
