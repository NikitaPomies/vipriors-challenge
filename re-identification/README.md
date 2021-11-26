# Re-Identification task for the VIPriors Challenge

Mantainer: Davide Zambrano from Synergy Sports (d.zambrano@sportradar.com)

**NOTE this code is based on Open-reid repo: https://github.com/Cysu/open-reid.git"**

_Open-ReID is a lightweight library of person re-identification for research
purpose. It aims to provide a uniform interface for different datasets, a full
set of models and evaluation metrics, as well as examples to reproduce (near)
state-of-the-art results._

We want to thank the authors for providing this tool. This version applies some changes to the original code to specifically adapt it to the VIPrior Challenge on Person Re-Identification. 

We present the "Visual Inductive Priors for Data-Efficient Computer Vision" challenge. This year, we offer five challenges, where models are to be trained from scratch in a data-deficient setting.

## Installation

**Note that the file ```setup.py``` specifies the libraries version to use to run the code.**

Install [PyTorch](http://pytorch.org/). 

```shell
git clone https://github.com/VIPriors/vipriors-challenges-toolkit
cd vipriors-challenges-toolkit/re-identification
pip install -e .
```

## Examples

```shell
python baseline/synergyreid_baseline.py -b 64 -j 2 -a resnet50 --logs-dir logs/synergy-reid/
```

This is just a quick example.

## Data

Person re-identification data are provided by [Synergy Sports](ttps://synergysports.com). Data come from short sequences of basketball games, each sequence is composed by 20 frames. For the validation and test sets, the query images are persons taken at the first frame, while the gallery images are identities taken from the 2nd to the last frame.

The idea behind the baseline is to provide a quick introduction to how to handle the re-id data. Specifically, attention should be put on the dataset creation and the dataloaders.
The data files are provided under ```baseline/data/synergyreid/raw/synergyreid_data.zip```.

The baseline code extracts the raw files in the same directory and prepares the splits to use for training, validation and test.

Specifically the dataset is divided as:

```shell
SynergyReID dataset loaded
  subset      | # ids | # images
  ---------------------------
  train       |   436 |     8569
  query val   |    50 |       50
  gallery val |    50 |      910
  trainval    |   486 |     9529
  ---------------------------
  query test  |   468 |      468
  gallery test |  8703 |     8703
```

Train and validation identities can be merged (to improve performance) using the flag ```--combine-trainval```.

The image filename is divided in three numbers: the first one is the person-id; the second one is the sequence where the image was taken; and the third one is the frame number.

The validation-set is divided in query and gallery to match the test-set format. With the flag ```--evaluate``` the distance matrix for the validation set is also saved.
The identities of the gallery are NOT provided; gallery ids are just random.

## Submission

You need to submit a .csv file as the pairwise distance matrix of size (m+1) x (n+1), where m is the number of querie images and n is the number of gallery images. The first row and the first column are the query and gallery ids respectively.
Query ids have to be orderd. Please check the example provided by the dataset loader in the baseline ```baseline/synergyreid_baseline.py```.
