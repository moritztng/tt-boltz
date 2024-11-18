import pickle
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import click
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm

from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml
from boltz.data.types import MSA, Manifest, Record
from boltz.data.write.writer import BoltzWriter
from boltz.model.model import Boltz1

CCD_URL = "https://www.dropbox.com/scl/fi/h4mjhcbhzzkkj4piu1k6x/ccd.pkl?rlkey=p43trjrs9ots4qk84ygk24seu&st=bymcsoqe&dl=1"
MODEL_URL = "https://www.dropbox.com/scl/fi/8qo9aryyttzp97z74dchn/boltz1.ckpt?rlkey=jvxl2jsn0kajnyfmesbj4lb89&st=dipi1sbw&dl=1"


@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


def download(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Warn user, just in case
    click.echo(
        f"Downloading data and model to {cache}. "
        "You may change this by setting the --cache flag."
    )

    # Download CCD, while capturing the output
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        click.echo(f"Downloading CCD to {ccd}.")
        urllib.request.urlretrieve(CCD_URL, str(ccd))  # noqa: S310

    # Download model
    model = cache / "boltz1.ckpt"
    if not model.exists():
        click.echo(f"Downloading model to {model}")
        urllib.request.urlretrieve(MODEL_URL, str(model))  # noqa: S310


def check_inputs(
    data: Path,
    outdir: Path,
    override: bool = False,
) -> list[Path]:
    """Check the input data and output directory.

    If the input data is a directory, it will be expanded
    to all files in this directory. Then, we check if there
    are any existing predictions and remove them from the
    list of input data, unless the override flag is set.

    Parameters
    ----------
    data : Path
        The input data.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    list[Path]
        The list of input data.

    """
    click.echo("Checking input data.")

    # Check if data is a directory
    if data.is_dir():
        data = list(data.glob("*"))
        data = [d for d in data if d.suffix in [".fasta", ".yaml"]]
    else:
        data = [data]

    # Check if existing predictions are found
    existing = (outdir / "predictions").rglob("*")
    existing = {e.name for e in existing if e.is_dir()}

    # Remove them from the input data
    if existing and not override:
        data = [d for d in data if d.stem not in existing]
        msg = "Found existing predictions, skipping and running only the missing ones."
        click.echo(msg)
    elif existing and override:
        msg = "Found existing predictions, will override."
        click.echo(msg)

    return data


def process_inputs(
    data: list[Path],
    out_dir: Path,
    ccd: dict[str, Mol],
    max_msa_seqs: int = 4096,
) -> BoltzProcessedInput:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    ccd : dict[str, Mol]
        The CCD dictionary.
    max_msa_seqs : int, optional
        Max number of MSA seuqneces, by default 4096.

    Returns
    -------
    BoltzProcessedInput
        The processed input data.

    """
    click.echo("Processing input data.")

    # Create output directories
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)

    # Parse input data
    records: list[Record] = []
    for path in tqdm(data):
        # Parse data
        if path.suffix == ".fasta":
            target = parse_fasta(path, ccd)
        elif path.suffix == ".yaml":
            target = parse_yaml(path, ccd)
        else:
            msg = f"Unable to parse filetype {path.suffix}"
            raise RuntimeError(msg)

        # Keep record
        records.append(target.record)

        # Dump structure
        struct_path = structure_dir / f"{target.record.id}.npz"
        target.structure.dump(struct_path)

    # Parse MSA data
    msas = {chain.msa_id for r in records for chain in r.chains if chain.msa_id != -1}
    msa_id_map = {}
    for msa_idx, msa_id in enumerate(msas):
        # Check that raw MSA exists
        msa_path = Path(msa_id)
        if not msa_path.exists():
            msg = f"MSA file {msa_path} not found."
            raise FileNotFoundError(msg)

        # Dump processed MSA
        processed = processed_msa_dir / f"{msa_idx}.npz"
        msa_id_map[msa_id] = msa_idx
        if not processed.exists():
            msa: MSA = parse_a3m(
                msa_path,
                taxonomy=None,
                max_seqs=max_msa_seqs,
            )
            msa.dump(processed)

    # Modify records to point to processed MSA
    for record in records:
        for c in record.chains:
            if c.msa_id != -1 and c.msa_id in msa_id_map:
                c.msa_id = msa_id_map[c.msa_id]

    # Dump manifest
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")

    return BoltzProcessedInput(
        manifest=manifest,
        targets_dir=structure_dir,
        msa_dir=processed_msa_dir,
    )


@click.group()
def cli() -> None:
    """Boltz1."""
    return


@cli.command()
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    help="The path where to save the predictions.",
    default="./",
)
@click.option(
    "--cache",
    type=click.Path(exists=False),
    help="The directory where to download the data and model. Default is ~/.boltz.",
    default="~/.boltz",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--devices",
    type=int,
    help="The number of devices to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--accelerator",
    type=click.Choice(["gpu", "cpu", "tpu"]),
    help="The accelerator to use for prediction. Default is gpu.",
    default="gpu",
)
@click.option(
    "--recycling_steps",
    type=int,
    help="The number of recycling steps to use for prediction. Default is 3.",
    default=3,
)
@click.option(
    "--sampling_steps",
    type=int,
    help="The number of sampling steps to use for prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples",
    type=int,
    help="The number of diffusion samples to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--output_format",
    type=click.Choice(["pdb", "mmcif"]),
    help="The output format to use for the predictions. Default is mmcif.",
    default="mmcif",
)
@click.option(
    "--num_workers",
    type=int,
    help="The number of dataloader workers to use for prediction. Default is 2.",
    default=2,
)
@click.option(
    "--override",
    is_flag=True,
    help="Whether to override existing found predictions. Default is False.",
)
def predict(
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
) -> None:
    """Run predictions with Boltz-1."""
    # If cpu, write a friendly warning
    if accelerator == "cpu":
        msg = "Running on CPU, this will be slow. Consider using a GPU."
        click.echo(msg)

    # Set no grad
    torch.set_grad_enabled(False)

    # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Create output directories
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
    download(cache)

    # Load CCD
    ccd_path = cache / "ccd.pkl"
    with ccd_path.open("rb") as file:
        ccd = pickle.load(file)  # noqa: S301

    # Set checkpoint
    if checkpoint is None:
        checkpoint = cache / "boltz1.ckpt"

    # Check if data is a directory
    data = check_inputs(data, out_dir, override)
    processed = process_inputs(data, out_dir, ccd)

    # Create data module
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=num_workers,
    )

    # Load model
    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
    }
    model_module: Boltz1 = Boltz1.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(BoltzDiffusionParams()),
    )
    model_module.eval()

    # Create prediction writer
    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format=output_format,
    )

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        strategy = DDPStrategy()

    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=32,
    )

    # Compute predictions
    trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=False,
    )


if __name__ == "__main__":
    cli()
