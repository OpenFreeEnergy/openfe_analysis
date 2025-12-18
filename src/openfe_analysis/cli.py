import json
import pathlib

import click

from . import rmsd


@click.group()
def cli():
    pass


@cli.command(name="RFE_analysis")
@click.option(
    "--pdb",
    type=click.Path(exists=True, readable=True, dir_okay=False, path_type=pathlib.Path),
    required=True,
    help="Path to the topology PDB file.",
)
@click.option(
    "--nc",
    type=click.Path(exists=True, readable=True, dir_okay=False, path_type=pathlib.Path),
    required=True,
    help="Path to the NetCDF trajectory file.",
)
@click.option(
    "--output",
    type=click.Path(writable=True, dir_okay=False, path_type=pathlib.Path),
    required=True,
    help="Path to save the JSON results.",
)
def rfe_analysis(pdb: pathlib.Path, nc: pathlib.Path, output: pathlib.Path):
    """
    Perform RMSD analysis for an RBFE simulation.

    Arguments:
        pdb: path to the topology PDB file.
        nc: path to the trajectory file (NetCDF format).
        output: path to save the JSON results.
    """
    # Run RMSD analysis
    data = rmsd.gather_rms_data(pdb, nc)

    # Write results
    with output.open("w") as f:
        json.dump(data, f, indent=2)
