import json
import pathlib

import click

from . import rmsd


@click.group()
def cli():
    pass


@cli.command(name="RFE_analysis")
@click.argument(
    "loc",
    type=click.Path(
        exists=True, readable=True, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
)
@click.argument("output", type=click.Path(writable=True, dir_okay=False, path_type=pathlib.Path))
def rfe_analysis(loc, output):
    pdb = loc / "hybrid_system.pdb"
    trj = loc / "simulation.nc"

    data = rmsd.gather_rms_data(pdb, trj)

    with click.open_file(output, "w") as f:
        f.write(json.dumps(data))
