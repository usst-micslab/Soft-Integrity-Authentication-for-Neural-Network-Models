# Experiment setup

## Environment

The checked environment used Python 3.10, PyTorch 2.7.1 and torchvision 0.22.1. Install dependencies from the repository root with `python -m pip install -r requirements.txt`.

## Authentication with existing fingerprints

The paper evaluates CIFAR-10, GTSRB and Tiny-ImageNet. The default authentication command uses CIFAR-10 with ResNet-18. Prepare the model state dictionary and the corresponding fingerprint images from the same experiment. Fingerprints are stored as an ImageFolder dataset with one or more class subdirectories.

```text
repository/
  model/
    test_model/resnet18_cifar.pth
  train_code/
    semi-fragile_samples_cifar/resnet18/3/<fingerprint images>.png
```

Run from `train_code/`.

```bash
python SIA.py --dataset cifar10 --architecture resnet18 --target-label 3 --limit 3
```

Leave out `--limit` to use all available fingerprints. `SIA_MODEL_ROOT` can point to the lab's external `model/` directory. Use `--samples-root` to supply the exact ImageFolder path. The output reports label agreement as counts and percentages. Dynamic quantization runs on CPU.

## Evaluate specified modified checkpoints

For the archived CIFAR-10 AlexNet experiment, place the baseline and matching modified checkpoints in the following folders under `model/`. Place its fingerprints under `train_code/semi-fragile_samples_cifar/alexnet/3/`.

```text
test_model/alexnet_cifar.pth
fintuned_model/alexnet_cifar_5.pth
pruning_model/alexnet_cifar_Conv1_0.1.pth
backdoor_model/alexnet_cifar_badnet_1.pth
```

```bash
python SIA.py --dataset cifar10 --architecture alexnet --finetune-epochs 5 --pruning-rates 0.1 --badnet-runs 1
```

Supply the checkpoint for every requested experiment. The backdoor loader also accepts the original nested layout under `backdoor_model/<architecture>/<dataset>/`.

## Select and generate fingerprints

Raw classifier data is needed for seed selection. Prepare the [CIFAR-10 Python archive](https://www.cs.toronto.edu/~kriz/cifar.html) under `data/cifar10/cifar-10-batches-py/`. The other paper branches use the [GTSRB training images](https://benchmark.ini.rub.de/gtsrb_dataset.html) as a class-folder dataset at `data/gtsrb/train/` or the [Tiny-ImageNet archive](http://cs231n.stanford.edu/tiny-imagenet-200.zip) at `data/tiny-imagenet-200/` with its original `train/`, `val/` and label files. Set `SIA_DATA_ROOT` to use another data directory.

Choose the same dataset and architecture in `Samples_selection.py`, `Samples_generation.py` and `MTI.py` before running their legacy experiment configurations. `Samples_selection.py` writes seeds to `selected_samples_<dataset>/<architecture>/<class>/`. For generation, supply the trained base, benign and six backdoor variants selected by `Samples_generation.py` and the corresponding `classifier/Map_*.pth` projection head. `Samples_generation.py` reads the selected seeds and writes fingerprints under `semi-fragile_samples_<dataset>/`. For modification type identification, `MTI.py` uses the matching `finetuning_classifier/*_fine-tuned_v1.pth` and the requested variant checkpoints. The packaged projection-head loader accepts the archived `ProjectionHead` checkpoint format.

See the [validation record](VALIDATION.md).
