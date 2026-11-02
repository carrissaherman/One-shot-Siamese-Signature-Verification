# One-Shot Siamese Signature Verification

Code for the paper **"Optimizing Siamese Networks for One-Shot Signature Verification: Contrastive vs Triplet Loss"** (Herman, Gunawan, Tedjasulaksana). [Paper link]

> Results in the paper correspond to tag [`paper-v1`](../../releases/tag/paper-v1).

## Overview
- **Task:** decide whether a signature is genuine or a skilled forgery, using a single reference signature per writer.
- **Privacy:** reference signatures are stored only as 64-d L2-normalized embeddings, never as raw images.
- **Model:** ImageNet-pretrained ResNet18 backbone, a 512→256→128→64 projection head, and Euclidean distance.
- **Compared:** contrastive loss (pairs) vs triplet loss (anchor/positive/negative).

## Data and evaluation
- CEDAR dataset: 55 writers, each with 24 genuine and 24 forged signatures.
- **Writer-disjoint split:** 45 writers for training, 10 unseen writers for testing, so there is no style leakage.

## Results (test writers)
| Metric | Contrastive | Triplet |
|---|---|---|
| Accuracy | **96.04%** | 65.00% |
| FAR | 4.58% | 35.00% |
| FRR | 3.33% | 35.00% |
| Mean genuine distance | 0.087 | 0.679 |
| Mean forgery distance | 1.025 | 1.011 |

**Why triplet loss failed:** with no hard-negative mining and little data, most triplets were "easy" ones that produced near-zero loss and therefore no gradient signal.

## Known limitations
- The decision threshold was selected on the test writers, so the reported accuracy is optimistic. A validation-based re-evaluation is planned.
- Only 10 test writers.

## Run
Open `One-shot signature verification model.ipynb` in Colab. The CEDAR dataset is available for academic use upon request from CEDAR, SUNY Buffalo.
