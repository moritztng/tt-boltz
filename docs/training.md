# Training

## Download processed data

Instructions on how to download the processed dataset for training are coming soon, we are currently uploading the data to sharable storage and will update this page when ready.

## Modify the configuration file

The training script requires a configuration file to run. This file specifies the paths to the data, the output directory, and other parameters of the data, model and training process. 

We provide under `scripts/train/configs` a template configuration file analogous to the one we used for training the structure model (`structure.yaml`) and the confidence model (`confidence.yaml`).

The following are the main parameters that you should modify in the configuration file to get the structure model to train:

```yaml
trainer:
  devices: 1

output: SET_PATH_HERE                 # Path to the output directory  
resume: PATH_TO_CHECKPOINT_FILE       # Path to a checkpoint file to resume training from if any null otherwise

data:
  datasets:
    - _target_: boltz.data.module.training.DatasetConfig
      target_dir: PATH_TO_TARGETS_DIR       # Path to the directory containing the processed structure files
      msa_dir: PATH_TO_MSA_DIR              # Path to the directory containing the processed MSA files

  symmetries: PATH_TO_SYMMETRY_FILE      # Path to the file containing molecule the symmetry information
  max_tokens: 512                        # Maximum number of tokens in the input sequence
  max_atoms: 4608                        # Maximum number of atoms in the input structure
```

`max_tokens` and `max_atoms` are the maximum number of tokens and atoms in the crop. Depending on the size of the GPUs you are using (as well as the training speed desired), you may want to adjust these values. Other recommended values are 256 and 2304, or 384 and 3456 respectively.

## Run the training script

Before running the full training, we recommend using the debug flag. This turns off DDP (sets single device) and sets `num_workers` to 0 so everything is in a single process, as well as disabling wandb:

    python scripts/train/train.py scripts/train/configs/structure.yaml debug=1

Once that seems to run okay, you can kill it and launch the training run:

    python scripts/train/train.py scripts/train/configs/structure.yaml

We also provide a different configuration file to train the confidence model:

    python scripts/train/train.py scripts/train/configs/confidence.yaml
