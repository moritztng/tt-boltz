![](banner.png)

# TT-Boltz

[Original Repo](https://github.com/jwohlwend/boltz) | [Boltz-1 Paper](https://doi.org/10.1101/2024.11.19.624167) | [Boltz-2 Paper](https://doi.org/10.1101/2025.06.14.659707)

TT-Boltz is the Boltz-2 implementation for inference on a single Tenstorrent Blackhole or Wormhole.

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
When following the [Tenstorrent Installation Guide](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md), you **must** use a different clone command to checkout the `boltz` branch:

**❌ Do NOT use the standard clone command from the installation guide:**
```bash
git clone https://github.com/tenstorrent/tt-metal.git --recurse-submodules
```

**✅ Instead, use this command to clone the `boltz` branch:**
```bash
git clone https://github.com/tenstorrent/tt-metal.git --branch boltz --recurse-submodules
```

Then continue with the rest of the installation guide.

### Install TT-NN into the Virtual Environment
From the `tt-metal` directory:
```bash
pip install -e .
```

### Install TT-Boltz into the Virtual Environment
From the `tt-boltz` directory:
```bash
pip install -e .
```

## Basic Usage

### Structure Prediction

Predict protein structures with automatic MSA generation:

```bash
tt-boltz predict examples/prot.yaml --use_msa_server --override
```

**Key Options:**
- `--use_msa_server`: Automatically generate MSAs (required if no MSA provided)
- `--override`: Re-run from scratch, ignoring cached files
- `--accelerator=tenstorrent`: Use Tenstorrent hardware (default, or use `cpu`/`gpu`)

### Binding Affinity Prediction

Predict binding affinity for protein-ligand complexes:

```bash
tt-boltz predict examples/affinity.yaml --use_msa_server --override --affinity_mw_correction
```

The `--affinity_mw_correction` flag applies molecular weight correction for more accurate predictions.

### Input Format

Create a YAML file describing your complex:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ
  - ligand:
      id: B
      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'
properties:
  - affinity:
      binder: B
```

**Entity Types:**
- **Polymers**: `protein`, `dna`, `rna` — provide `sequence`
- **Ligands**: `ligand` — provide `smiles` or `ccd` code

**Multiple Identical Chains:**
```yaml
- protein:
    id: [A, B]  # Two identical chains
    sequence: ...
```

## Understanding Results

### Output Structure

```
boltz_results_prot/
├── predictions/
│   └── prot/
│       ├── prot_model_0.cif          # Predicted structure (ranked by confidence)
│       ├── confidence_prot_model_0.json
│       ├── affinity_prot.json        # (if affinity prediction enabled)
│       ├── pae_prot_model_0.npz      # Predicted aligned error
│       ├── pde_prot_model_0.npz      # Predicted distance error
│       └── plddt_prot_model_0.npz    # Per-residue confidence
└── processed/                         # Cached preprocessing data
```

### Confidence Scores

The `confidence_*.json` file contains:

```json
{
    "confidence_score": 0.84,      # Overall confidence (0-1, higher is better)
    "ptm": 0.84,                   # Predicted TM-score for complex
    "iptm": 0.82,                  # Interface TM-score
    "complex_plddt": 0.84,         # Average per-residue confidence
    "chains_ptm": {                # Per-chain TM-scores
        "0": 0.85,
        "1": 0.83
    }
}
```

**Interpretation:**
- Scores range 0-1 (higher = better)
- `confidence_score` = 0.8 × `complex_plddt` + 0.2 × `iptm`
- Models are ranked by `confidence_score`

### Affinity Predictions

The `affinity_*.json` file contains:

```json
{
    "affinity_pred_value": 2.47,           # log10(IC50) in μM
    "affinity_probability_binary": 0.41,   # Probability of binding (0-1)
    "affinity_pred_value1": 2.55,          # Individual model predictions
    "affinity_pred_value2": 2.19
}
```

**Two Different Metrics:**

1. **`affinity_probability_binary`** (0-1): Use for **hit discovery**
   - Probability that ligand is a binder
   - Higher = more likely to bind
   - Use to filter binders from decoys

2. **`affinity_pred_value`**: Use for **ligand optimization**
   - `log10(IC50)` where IC50 is in μM
   - Lower = stronger binding
   - Only compare between known active molecules
   - Examples:
     - `-3` = IC50 of 10⁻⁹ M (strong binder)
     - `0` = IC50 of 10⁻⁶ M (moderate binder)
     - `2` = IC50 of 10⁻⁴ M (weak binder)

**Convert to pIC50 (kcal/mol):**
```
pIC50 = (6 - affinity_pred_value) × 1.364
```

## Advanced Usage

### Input Format Details

#### Proteins with Custom MSA
```yaml
- protein:
    id: A
    sequence: MVTPEGNVSLVDES...
    msa: ./path/to/msa.a3m
```

#### Proteins with Modifications
```yaml
- protein:
    id: A
    sequence: MVTPEGNVSLVDES...
    modifications:
      - position: 5
        ccd: PTR  # Modified residue code
```

#### Ligands
```yaml
- ligand:
    id: B
    smiles: 'CC1=CC=CC=C1'  # SMILES string
    # OR
    ccd: ATP                # CCD code
```

#### Constraints

**Pocket Constraints** (binding site):
```yaml
constraints:
  - pocket:
      binder: B              # Ligand chain
      contacts: [[A, 10], [A, 11], [A, 12]]  # Binding site residues
      max_distance: 6.0      # Angstroms (4-20A, default 6A)
      force: false           # Use potential to enforce (default: false)
```

**Contact Constraints:**
```yaml
constraints:
  - contact:
      token1: [A, 10]
      token2: [A, 50]
      max_distance: 8.0
      force: false
```

#### Templates

Use experimental structures as templates:

```yaml
templates:
  - cif: ./template.cif
    chain_id: A
    template_id: A
    force: true              # Enforce template alignment
    threshold: 2.0           # Max deviation in Angstroms
```

### Command-Line Options

**Common Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--out_dir` | `./` | Output directory |
| `--cache` | `~/.boltz` | Model cache directory |
| `--accelerator` | `tenstorrent` | `tenstorrent`, `cpu`, or `gpu` |
| `--recycling_steps` | `3` | Number of recycling iterations |
| `--sampling_steps` | `200` | Diffusion sampling steps |
| `--diffusion_samples` | `1` | Number of structure samples |
| `--output_format` | `cif` | `cif` or `pdb` |
| `--override` | `False` | Re-run from scratch |
| `--use_msa_server` | `False` | Auto-generate MSAs |
| `--use_potentials` | `False` | Apply physical constraints |
| `--affinity_mw_correction` | `False` | Apply MW correction to affinity |

**Affinity-Specific Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--sampling_steps_affinity` | `200` | Sampling steps for affinity |
| `--diffusion_samples_affinity` | `5` | Number of affinity samples |

**MSA Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--msa_server_url` | `https://api.colabfold.com` | MSA server URL |
| `--msa_pairing_strategy` | `greedy` | `greedy` or `complete` |
| `--max_msa_seqs` | `8192` | Maximum MSA sequences |
| `--subsample_msa` | `False` | Subsample MSA |
| `--num_subsampled_msa` | `1024` | Number of subsampled sequences |

### MSA Server Authentication

**Basic Authentication:**
```bash
export BOLTZ_MSA_USERNAME=myuser
export BOLTZ_MSA_PASSWORD=mypassword
tt-boltz predict ... --use_msa_server
```

**API Key Authentication:**
```bash
export MSA_API_KEY_VALUE=your-api-key
tt-boltz predict ... --use_msa_server --api_key_header X-API-Key
```

## Performance

Runtime for a 686 amino acid protein:

| Hardware | Time |
|----------|------|
| AMD Ryzen 5 8600G | ~45 min |
| Nvidia T4 | ~9 min |
| Tenstorrent Blackhole p150 | ~1 min |
| Nvidia RTX 5090 | ~1 min |

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
  year={2022}
}
```

## License

MIT License
