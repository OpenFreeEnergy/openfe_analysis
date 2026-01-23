import netCDF4 as nc
import numpy as np
import pytest
from numpy.testing import assert_allclose

from openfe_analysis.rmsd import gather_rms_data


@pytest.mark.flaky(reruns=3)
def test_gather_rms_data_regression(simulation_nc, hybrid_system_pdb):
    output = gather_rms_data(
        hybrid_system_pdb,
        simulation_nc,
        skip=100,
    )

    assert_allclose(output["time(ps)"], [0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    assert len(output["protein_RMSD"]) == 3
    assert_allclose(
        output["protein_RMSD"][0],
        [0.0, 1.003, 1.276, 1.263, 1.516, 1.251],
        rtol=1e-3,
    )
    assert len(output["ligand_RMSD"]) == 3
    assert_allclose(
        output["ligand_RMSD"][0],
        [0.0, 0.9094, 1.0398, 0.9774, 1.9108, 1.2149],
        rtol=1e-3,
    )
    assert len(output["ligand_wander"]) == 3
    assert_allclose(
        output["ligand_wander"][0],
        [0.0, 0.5458, 0.8364, 0.4914, 1.1939, 0.7587],
        rtol=1e-3,
    )
    assert len(output["protein_2D_RMSD"]) == 3
    # 15 entries because 6 * 6 frames // 2
    assert len(output["protein_2D_RMSD"][0]) == 15
    assert_allclose(
        output["protein_2D_RMSD"][0][:6],
        [1.0029, 1.2756, 1.2635, 1.5165, 1.2509, 1.0882],
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
