import click
import json
from os import path

from . import rmsd


@click.command
@click.argument('loc', type=click.Path(exists=True,
                                       readable=True,
                                       file_okay=False,
                                       dir_okay=True))
def main(loc):
    pdb = path.join(loc, "hybrid_system.pdb")
    trj = path.join(loc, "simulation.nc")

    #click.echo(f'hello there {pdb} {trj}')
    data = rmsd.gather_rms_data(pdb, trj)
    click.echo(json.dumps(data))
