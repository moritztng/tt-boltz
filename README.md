<div align="center">
  <div>&nbsp;</div>
  <img src="docs/boltz_title.png" width="400"/>

[Paper](https://doi.org/10.1101/2024.11.19.624167)
</div>


![](docs/boltz1_pred_figure.png)


## Introduction

TT-Boltz is the Boltz-1 fork that runs on a single Tenstorrent Wormhole n150 or n300.

Boltz-1 is the state-of-the-art open-source model to predict biomolecular structures containing combinations of proteins, RNA, DNA, and other molecules. It also supports modified residues, covalent ligands and glycans, as well as conditioning the prediction on specified interaction pockets or contacts. 

All the code and weights are provided under MIT license, making them freely available for both academic and commercial uses. For more information about the model, see our [technical report](https://doi.org/10.1101/2024.11.19.624167).

## Installation
### Clone
```bash
git clone https://github.com/moritztng/tt-boltz.git
cd tt-boltz
```
### Create Virtual Environment
```bash
python3 -m venv env
source env/bin/activate
```
### Build TT-Metal from Source
Don't install tt-nn with `./create_venv.sh`.

[Tenstorrent Installation Guide](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
### Install TT-NN
```bash
pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
pip install setuptools wheel
pip install -r <path-to-tt-metal-repo>/tt_metal/python_env/requirements-dev.txt
pip install <path-to-tt-metal-repo>
```
### Install TT-Boltz
```bash
pip install -e .
```
You can ignore the error about the pandas version.
## Inference

You can run inference using Boltz-1 with:

```
boltz predict input_path --use_msa_server --accelerator=tenstorrent
```

Pass `--accelerator=tenstorrent` to run on Tenstorrent Wormhole.

Boltz currently accepts three input formats:

1. Fasta file, for most use cases

2. A comprehensive YAML schema, for more complex use cases

3. A directory containing files of the above formats, for batched processing

To see all available options: `boltz predict --help` and for more information on these input formats, see our [prediction instructions](docs/prediction.md).

## Evaluation

To encourage reproducibility and facilitate comparison with other models, we provide the evaluation scripts and predictions for Boltz-1, Chai-1 and AlphaFold3 on our test benchmark dataset as well as CASP15. These datasets are created to contain biomolecules different from the training data and to benchmark the performance of these models we run them with the same input MSAs and same number  of recycling and diffusion steps. More details on these evaluations can be found in our [evaluation instructions](docs/evaluation.md).

![Test set evaluations](docs/plot_test.png)
![CASP15 set evaluations](docs/plot_casp.png)


## Training

If you're interested in retraining the model, see our [training instructions](docs/training.md).

## Community
- [I build in public on X](https://x.com/moritzthuening)
- [Boltz-1 Slack](https://join.slack.com/t/boltz-community/shared_invite/zt-2zj7e077b-D1R9S3JVOolhv_NaMELgjQ)
- [Tenstorrent Discord](https://discord.gg/tvhGzHQwaj)

## License

Our model and code are released under MIT License, and can be freely used for both academic and commercial purposes.


## Cite

If you use this code or the models in your research, please cite the following paper:

```bibtex
@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  year = {2024},
  doi = {10.1101/2024.11.19.624167},
  journal = {bioRxiv}
}
```

In addition if you use the automatic MSA generation, please cite:

```bibtex
@article{mirdita2022colabfold,
  title={ColabFold: making protein folding accessible to all},
  author={Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
  journal={Nature methods},
  year={2022},
}
```
