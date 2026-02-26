import MDAnalysis as mda
import numpy as np
import pytest
from rdkit.Chem import Lipinski

from openfe_analysis.reader import FEReader
from openfe_analysis.prolif import ProLIFAnalysis


def test_prolifanalysis_runs_vdwcontact(
    simulation_skipped_nc, hybrid_system_skipped_pdb
):
    """
    Test for identification of interactions
    """
    u = mda.Universe(
        hybrid_system_skipped_pdb, simulation_skipped_nc, format=FEReader, index=0
    )
    ligand_ag = u.select_atoms("resname UNK")

    analysis = ProLIFAnalysis(
        u, ligand_ag, interactions=["VdWContact"], guess_bonds=True
    )
    analysis.run(stop=5, step=1, n_jobs=1, progress=False)

    df = analysis.to_dataframe(dtype=np.uint8)
    assert df.shape[0] == 5
    assert hasattr(analysis.fp, "ifp")
    assert len(analysis.fp.ifp) == 5

    # Check AnalysisBase
    assert hasattr(analysis, "results")
    assert hasattr(analysis.results, "ifp")
    assert analysis.results.ifp is analysis.fp.ifp
    assert len(analysis.results.ifp) == 5

    assert analysis.results.ifp_df is df

    # Ensure there is at least one detected interaction across all processed frames
    assert sum(len(v) for v in analysis.fp.ifp.values()) > 0


def test_guess_bonds_enables_protein_chemistry(
    simulation_skipped_nc, hybrid_system_skipped_pdb
):
    """
    Test for protein connectivity
    """
    u = mda.Universe(
        hybrid_system_skipped_pdb, simulation_skipped_nc, format=FEReader, index=0
    )
    ligand_ag = u.select_atoms("resname UNK")

    analysis = ProLIFAnalysis(
        u, ligand_ag, interactions=["VdWContact"], guess_bonds=True
    )

    # pick a residue from the pocket and check it has connectivity in RDKit
    u.trajectory[0]
    res_atoms = analysis.protein_ag.residues[0].atoms
    res_mol = res_atoms.convert_to("RDKIT", implicit_hydrogens=False)
    assert res_mol.GetNumBonds() > 0

    # ensure the protein donors/acceptors exist
    prot_mol = analysis.protein_ag.convert_to("RDKIT", implicit_hydrogens=False)
    assert Lipinski.NumHDonors(prot_mol) + Lipinski.NumHAcceptors(prot_mol) > 0
