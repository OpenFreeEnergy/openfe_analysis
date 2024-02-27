import click
from click.exceptions import BadOptionUsage
import json
import netCDF4 as nc
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


_statehelp = """\
"""


@cli.command(name='RFE_trjconv')
@click.argument("loc", type=click.Path(exists=True,
                                       readable=True,
                                       file_okay=False,
                                       dir_okay=True,
                                       path_type=pathlib.Path))
@click.argument('output', type=click.Path(writable=True,
                                        dir_okay=False,
                                        exists=False,
                                        path_type=pathlib.Path),
                        required=True)
@click.option('-s', '--state', required=True,
              help="either an integer (0 and -1 giving endstates) or 'all'")
def RFE_trjconv(loc, output, state):
    """Convert .nc trajectory files from RBFE to new format for a single state

    LOC is the directory where a simulation took place, it should contain the
    simulation.nc and hybrid_system.pdb files that were produced.

    OUTPUT is the name of the new trajectory file, e.g. "out.xtc".  Any file
    format supported by MDAnalysis can be specified, including XTC and DCD
    formats.

    The .nc trajectory file contains multiple states; a single state can be
    specified for output.  Negative indices are allowed and treated as in
    Python, therefore ``--state=0`` or ``--state=-1`` will produce trajectories
    of the two end states.

    If ``--state='all'`` is given, all states are outputted, and the output
    filename has the state number inserted before the file prefix,
    e.g. ``--output=traj.dcd`` would produce a files called ``traj_state0.dcd``
    etc.
    """
    import MDAnalysis as mda

    pdb = loc / "hybrid_system.pdb"
    trj = loc / "simulation.nc"

    ds = nc.Dataset(trj, mode='r')

    if state == 'all':
        # figure out how many states we need to output
        nstates = ds.dimensions['state'].size

        states = range(nstates)
        # turn out.dcd -> out_0.dcd
        outputs = [
            output.with_stem(output.stem + f'_state{i}')
            for i in range(nstates)
        ]
    else:
        try:
            states = [int(state)]
        except ValueError:
            raise BadOptionUsage(f"Invalid state specified: {state}")
        outputs = [output]

    for s, o in zip(states, outputs):
        u = mda.Universe(pdb, ds,
                         format=FEReader,
                         state_id=s)
        ag = u.atoms  # todo, atom selections would be here

        with mda.Writer(str(o), n_atoms=len(ag)) as w:
            for ts in tqdm.tqdm(u.trajectory):
                w.write(ag)
