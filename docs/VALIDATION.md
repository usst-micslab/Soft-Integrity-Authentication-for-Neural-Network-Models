# Validation record

Windows CPU with Python 3.10.20, PyTorch 2.7.1 and torchvision 0.22.1.

## Archived inputs

- Ran the `SIA.py` entry on three archived CIFAR-10 ResNet-18 fingerprints with the original and dynamically quantized models.
- Ran the CIFAR-10 AlexNet entry on three archived fingerprints with the original, dynamically quantized, five-epoch fine-tuned, 0.1-pruned and first BadNet checkpoints. The expected-label counts were 3/3 for the first four and 0/3 for the BadNet checkpoint. These counts describe the three-input check.

## Synthetic input

Loaded the archived AlexNet projection heads used by `Samples_generation.py` and `MTI.py` and passed a synthetic `(2, 10)` score matrix through each. Both produced finite `(2, 10)` outputs. This validates classifier deserialization and the forward tensor dimensions.

Full fingerprint optimization, modification type identification over all variants, and the paper's benchmark remain outside these checks.
