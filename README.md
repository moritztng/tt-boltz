
![](docs/boltz1_pred_figure.png)

# TT-Boltz
[Original Repo](https://github.com/jwohlwend/boltz) | [Boltz-1](https://doi.org/10.1101/2024.11.19.624167) | [Boltz-2](https://doi.org/10.1101/2025.06.14.659707)

## Introduction
TT-Boltz is the Boltz-2 fork that runs on a single Tenstorrent Blackhole or Wormhole.

Boltz is a family of models for biomolecular interaction prediction. Boltz-1 was the first fully open source model to approach AlphaFold3 accuracy. Our latest work Boltz-2 is a new biomolecular foundation model that goes beyond AlphaFold3 and Boltz-1 by jointly modeling complex structures and binding affinities, a critical component towards accurate molecular design. Boltz-2 is the first deep learning model to approach the accuracy of physics-based free-energy perturbation (FEP) methods, while running 1000x faster — making accurate in silico screening practical for early-stage drug discovery.

All the code and weights are provided under MIT license, making them freely available for both academic and commercial uses. For more information about the model, see the [Boltz-1](https://doi.org/10.1101/2024.11.19.624167) and [Boltz-2](https://doi.org/10.1101/2025.06.14.659707) technical reports.

For an intuitive understanding of AlphaFold 3, I recommend [The Illustrated AlphaFold](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold).

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
pip install setuptools wheel==0.45.1
pip install -r <path-to-tt-metal-repo>/tt_metal/python_env/requirements-dev.txt
pip install -e <path-to-tt-metal-repo>
```
### Install TT-Boltz
```bash
pip install -e .
```
You can ignore the version warnings.
## Inference

You can run inference using Boltz with:

```
boltz predict examples/prot.yaml --use_msa_server --override --accelerator=tenstorrent
```

Pass `--accelerator=tenstorrent` to run Boltz-2 on Tenstorrent Blackhole or Wormhole. Boltz-1 is not supported on Tenstorrent hardware anymore.

The input path should point to a YAML file, or a directory of YAML files for batched processing, describing the biomolecules you want to model and the properties you want to predict (e.g. affinity). To see all available options: `boltz predict --help` and for more information on these input formats, see our [prediction instructions](docs/prediction.md). By default, the `boltz` command will run the latest version of the model.

### Binding Affinity Prediction
There are two main predictions in the affinity output: `affinity_pred_value` and `affinity_probability_binary`. They are trained on largely different datasets, with different supervisions, and should be used in different contexts. The `affinity_probability_binary` field should be used to detect binders from decoys, for example in a hit-discovery stage. Its value ranges from 0 to 1 and represents the predicted probability that the ligand is a binder. The `affinity_pred_value` aims to measure the specific affinity of different binders and how this changes with small modifications of the molecule. This should be used in ligand optimization stages such as hit-to-lead and lead-optimization. It reports a binding affinity value as `log10(IC50)`, derived from an `IC50` measured in `μM`. More details on how to run affinity predictions and parse the output can be found in our [prediction instructions](docs/prediction.md).

## Authentication to MSA Server
When using the `--use_msa_server` option with a server that requires authentication, you can provide credentials in one of two ways. More information is available in our [prediction instructions](docs/prediction.md).
 
## Current Runtime
We didn't even start writing low-level code yet and Tenstorrent Blackhole was just released. I'm confident we'll get to 2 minutes. 
|Hardware|~Minutes|
|---|---|
|AMD Ryzen 5 8600G|45|
|Nvidia T4|9|
|Tenstorrent Wormhole n300|3|
|Tenstorrent Blackhole p150|1|
|Nvidia RTX 4090|1|
## Evaluation

⚠️ **Coming soon: updated evaluation code for Boltz-2!**

To encourage reproducibility and facilitate comparison with other models, on top of the existing Boltz-1 evaluation pipeline, we will soon provide the evaluation scripts and structural predictions for Boltz-2, Boltz-1, Chai-1 and AlphaFold3 on our test benchmark dataset, and our affinity predictions on the FEP+ benchmark, CASP16 and our MF-PCBA test set.

![Affinity test sets evaluations](docs/pearson_plot.png)
![Test set evaluations](docs/plot_test_boltz2.png)

## Community
- [I build in public on X](https://x.com/moritzthuening)
- [Boltz Slack](https://join.slack.com/t/boltz-community/shared_invite/zt-2zj7e077b-D1R9S3JVOolhv_NaMELgjQ)
- [Tenstorrent Discord](https://discord.gg/tvhGzHQwaj)

## Cite

If you use this code or the models in your research, please cite the following papers:

```bibtex
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Thaler, Stephan and Somnath, Vignesh Ram and Getz, Noah and Portnoi, Tally and Roy, Julien and Stark, Hannes and Kwabi-Addo, David and Beaini, Dominique and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}

@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Getz, Noah and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and Atkinson, Liam and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and Jaakkola, Tommi and Barzilay, Regina},
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
