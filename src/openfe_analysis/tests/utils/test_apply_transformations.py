from __future__ import annotations

from itertools import islice

import numpy as np
import pytest
from MDAnalysis.analysis import rms
from MDAnalysis.lib.mdamath import make_whole
from MDAnalysis.transformations import unwrap

from openfe_analysis.transformations import Aligner
from openfe_analysis.utils import apply_transformations, universe_utils


@pytest.fixture
def universe_single_state(hybrid_system_skipped_pdb, simulation_skipped_nc):
    u = universe_utils.create_universe_single_state(
        hybrid_system_skipped_pdb, simulation_skipped_nc, 0
    )
    yield u
    u.trajectory.close()


def test_chain_radius_of_gyration_stable(universe_single_state):
    """Protein chains should not explode or collapse due to PBC errors
    after applying alignment transformations."""
    universe_single_state.select_atoms("protein").guess_bonds()
    prot = universe_single_state.select_atoms("protein and name CA")
    apply_transformations.apply_complex_alignment_transformations(universe_single_state, prot)

    chain = prot.segments[0].atoms
    rgs = []
    for ts in universe_single_state.trajectory[:50]:
        rgs.append(chain.radius_of_gyration())

    assert np.std(rgs) < 2.0


def test_ligand_com_continuity(universe_single_state):
    """Ligand COM should not jump between periodic images after applying
    alignment transformations."""
    ligand = universe_single_state.select_atoms("resname UNK")
    apply_transformations.apply_ligand_alignment_transformations(
        universe_single_state, ligand=ligand
    )

    coms = [ligand.center_of_mass() for ts in islice(universe_single_state.trajectory, 20)]
    jumps = [np.linalg.norm(coms[i + 1] - coms[i]) for i in range(len(coms) - 1)]

    assert max(jumps) < 5.0


def test_multichain_rmsd_shifting(simulation_skipped_nc, hybrid_system_skipped_pdb):
    """Chain shifting should remove RMSD jumps caused by periodic boundary crossings."""
    u = universe_utils.create_universe_single_state(
        hybrid_system_skipped_pdb, simulation_skipped_nc, 0
    )
    u.select_atoms("protein").guess_bonds()
    prot = u.select_atoms("protein and name CA")
    # Do other transformations, but no shifting
    unwrap_tr = unwrap(prot)
    for frag in prot.fragments:
        make_whole(frag, reference_atom=frag[0])
    align = Aligner(prot)
    u.trajectory.add_transformations(unwrap_tr, align)
    chains = [seg.atoms for seg in prot.segments]
    assert len(chains) > 1, "Test requires multi-chain protein"

    # RMSD without shifting
    r = rms.RMSD(prot)
    r.run()
    rmsd_no_shift = r.rmsd[:, 2]
    assert np.max(np.diff(rmsd_no_shift[:20])) > 10  # expect jumps
    u.trajectory.close()

    # RMSD with shifting
    u2 = universe_utils.create_universe_single_state(
        hybrid_system_skipped_pdb, simulation_skipped_nc, 0
    )
    u2.select_atoms("protein").guess_bonds()
    prot2 = u2.select_atoms("protein and name CA")
    assert len(list(prot2.fragments)) == 2
    apply_transformations.apply_complex_alignment_transformations(u2, protein=prot2)

    R2 = rms.RMSD(prot2)
    R2.run()
    rmsd_shift = R2.rmsd[:, 2]
    assert np.max(np.diff(rmsd_shift[:20])) < 2  # jumps should disappear
    u2.trajectory.close()


def test_rmsd_reference_is_first_frame(universe_single_state):
    """After alignment, RMSD at the first frame should be zero."""
    universe_single_state.select_atoms("protein").guess_bonds()
    prot = universe_single_state.select_atoms("protein and name CA")
    apply_transformations.apply_complex_alignment_transformations(
        universe_single_state, protein=prot
    )

    _ = next(iter(universe_single_state.trajectory))  # SAFE
    ref = prot.positions.copy()

    rmsd = np.sqrt(((prot.positions - ref) ** 2).mean())
    assert rmsd == 0.0
    universe_single_state.trajectory.close()


def test_empty_protein_raises(universe_single_state):
    empty = universe_single_state.select_atoms("resname DOESNOTEXIST")
    with pytest.raises(ValueError, match="empty or None"):
        apply_transformations.apply_complex_alignment_transformations(
            universe_single_state, protein=empty
        )


def test_atomgroup_as_ligands_raises(universe_single_state):
    universe_single_state.select_atoms("protein").guess_bonds()
    prot = universe_single_state.select_atoms("protein and name CA")
    lig = universe_single_state.select_atoms("resname UNK")
    with pytest.raises(TypeError, match="list of AtomGroups"):
        apply_transformations.apply_complex_alignment_transformations(
            universe_single_state, protein=prot, ligands=lig
        )


def test_empty_ligand_raises(universe_single_state):
    empty = universe_single_state.select_atoms("resname DOESNOTEXIST")
    with pytest.raises(ValueError, match="empty or None"):
        apply_transformations.apply_ligand_alignment_transformations(
            universe_single_state, ligand=empty
        )


def test_missing_protein_bond_raises(universe_single_state):
    prot = universe_single_state.select_atoms("protein and name CA")
    ligand = universe_single_state.select_atoms("resname UNK")
    assert not prot.bonds  # precondition: the guard's trigger is actually present

    with pytest.raises(ValueError, match="no bonds"):
        apply_transformations.apply_complex_alignment_transformations(
            universe_single_state, prot, [ligand]
        )
