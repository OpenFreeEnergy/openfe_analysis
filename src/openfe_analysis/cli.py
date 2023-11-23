import click
import json
import pathlib
import tqdm

from . import rmsd
from .reader import FEReader


@click.group()
def cli():
    pass


@cli.command(name="RFE_analysis")
@click.argument('loc', type=click.Path(exists=True,
                                       readable=True,
                                       file_okay=False,
                                       dir_okay=True,
                                       path_type=pathlib.Path))
@click.argument('output', type=click.Path(writable=True,
                                          dir_okay=False,
                                          path_type=pathlib.Path))
def rfe_analysis(loc, output):
    """Perform structural analysis on OpenMM RFE simulation"""
    pdb = loc / "hybrid_system.pdb"
    trj = loc / "simulation.nc"

    data = rmsd.gather_rms_data(pdb, trj)

    with click.open_file(output, 'w') as f:
        f.write(json.dumps(data))


@cli.command(name='trjconv')
@click.argument("loc", type=click.Path(exists=True,
                                       readable=True,
                                       file_okay=False,
                                       dir_okay=True,
                                       path_type=pathlib.Path))
@click.argument('output', type=click.Path(writable=True,
                                          dir_okay=False,
                                          exists=False,
                                          path_type=pathlib.Path))
@click.option('-s', '--state', type=int, required=True)
def trjconv(loc, output, state):
    """Convert .nc trajectory files to new format for a single state

    LOC is the directory where a simulation took place, it should contain the
    simulation.nc and hybrid_system.pdb files that were produced.

    OUTPUT is the name of the new trajectory file, e.g. "out.xtc".  Any file
    format supported by MDAnalysis can be specified, including XTC and DCD
    formats.

    The .nc trajectory file contains multiple states; a single state must be
    specified for output.  Negative indices are allowed and treated as in
    Python, therefore "--state=0" or "--state=-1" will produce trajectories of
    the two end states.
    """
    import MDAnalysis as mda

    pdb = loc / "hybrid_system.pdb"
    trj = loc / "simulation.nc"

    u = mda.Universe(pdb, trj,
                     format=FEReader,
                     state_id=state)
    ag = u.atoms  # todo, selections would be here

    with mda.Writer(str(output), n_atoms=len(ag)) as w:
        for ts in tqdm.tqdm(u.trajectory):
            w.write(ag)
