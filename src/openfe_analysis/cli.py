import click
import json
import pathlib

from . import rmsd


@click.group()
def cli():
    pass


@cli.command(name="RFE_analysis")
@click.argument('loc', type=click.Path(exists=True,
                                       readable=True,
                                       file_okay=False,
                                       dir_okay=True,
                                       path_type=pathlib.Path))
def rfe_analysis(loc):
    pdb = loc / "hybrid_system.pdb"
    trj = loc / "simulation.nc"

    data = rmsd.gather_rms_data(pdb, trj)
    click.echo(json.dumps(data))
