# Siam Mesh R-CNN
This repository contains the implementation for the siamese mesh tracker.
Implemented for the master's thesis "Object Tracking Using a Mesh-Convolutional network".

For all scripts provided, the path to the dataset might need adaption.

Also the pre-trained backbone and RPN network, provided by [Tensorpack](https://github.com/tensorpack/tensorpack) (visited 10.2.2022), needs to be downloaded and placed in, it is available [here](http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R50FPN4xGNCasAug.npz) (visited 10.2.2022).

# Requirements
To set up an environment for training and evaluating Siam Mesh RCNN, we recommend to use a virtual environment.

The tested environment uses Python 3.6.9.
Since some packages need to be built and installed from source, the following libraries must be installed
```
apt install -y screen build-essential \
    libssl-dev libffi-dev libxml2-dev libxslt1-dev \
    zlib1g-dev libosmesa6-dev libgl1-mesa-dev \
    libglu1-mesa-dev freeglut3-dev libboost-dev
```
The following python packages can be installed directly:
```
pip install scikit-learn opendr readchar openmesh \
    msgpack msgpack-numpy psutil tabulate  opencv-python \
    xmltodict wget shapely pycocotools
```
The [Perceiving Systems Mesh Package](https://github.com/MPI-IS/mesh) (visited 10.02.2022) needs to be manually built and installed.
```
git clone https://github.com/MPI-IS/mesh.git
cd mesh && make all && cd ..
```
Further, Tensorpack, and the [GOT10k Python Toolkit](https://github.com/pvoigtlaender/got10k-toolkit) (visited 10.02.2022), forked by Voigtlaender et al. need to cloned and put on the PYTHONPATH.
```
git clone https://github.com/pvoigtlaender/got10k-toolkit.git
git clone https://github.com/tensorpack/tensorpack.git
```
The paths need to be adapted, this also adds the Siam Mesh R-CNN to the python path, for all scripts to work properly.
```
echo 'export PYTHONPATH=${PYTHONPATH}:/PATH/TO/got10k-toolkit/:/PATH/TO/tensorpack/:/PATH/TO/conv-track' >> ~/.bashrc
```


# Data
This section will explain data preparation.

## Dataset
The dataset used for training and evaluating the model is the Chokepoint dataset by Wong et al. released by NICTA, available [here](http://arma.sourceforge.net/chokepoint/) (visited 10.2.2022).

## Bounding Boxes
As bounding box annotations for the faces in the dataset the annotations provided by Alver et al. need to be supplied, available [here](https://github.com/alversafa/chokepoint-bbs) (visited 10.2.2022).

The bounding box annotations need to be converted to sequences that can be consumed for training.
```
./annotation_to_sequences.sh
```

## Mesh Ground Truth
For mesh ground truth generation execute the following:

```
./generate_mesh_ground_truth.sh
```

## FPN Feature Extraction
FPN feature extraction, for training FPN2Mesh:
```
./fpn_feature_extraction.sh
```

# Training

Several utilites exist for training the model, since some parts are pre-trained for mesh extraction these have to be executed first.

## FPN2Mesh
Training FPN2Mesh
```
./train_fpn2mesh.sh
```
It produces 4 FPN2Mesh networks. Two using the spiral convolution operator, once trained with L1 vertex loss only and once with L1 vertex loss and edge loss.
Two using the chebyshev convolution, once trained with L1 vertex loss and once with L1 vertex loss and edge loss.

## Mesh Autoencoder
To train the mesh autoencoder, execute:
```
./train_autoencoder.sh
```
It produces 4 autoencoders. Two using the spiral convolution operator, once trained with L1 vertex loss only and once with L1 vertex loss and edge loss.
Two using the chebyshev convolution, once trained with L1 vertex loss and once with L1 vertex loss and edge loss.

## Siam Mesh R-CNN
Training the Siam Mesh RCNN. In order to train the Siam Mesh R-CNN, 
the mesh autoencoder for the mesh encoding and the  FPN2Mesh network need to be trained first.
For training, the weights for the base model and the FPN2Mesh model as well as the Mesh2Feature model need to be provided.

Before the trained CoMA and FPN2Mesh weights can be used for Siam Mesh R-CNN, the checkpoints need to be converted to dictionaries.
For this purpose use the following script:

```
./convert_checkpoint_to_dict.sh
```

With these converted dictionaries the Siam Mesh R-CNN can be trained with:
```
./train_siam_mesh_rcnn.sh
./train_siam_mesh_rcnn_lc.sh
```

## Siam R-CNN baseline
Training the Siam RCNN baseline model

```
./train_siam_rcnn_baseline.sh
```

# Evaluation
The following scripts are utilities to evaluate separate parts of Siam Mesh R-CNN,
as well as the complete network.

## FPN2Mesh
For evaluation of the FPN2Mesh, first perform the predictions then calculate the errors with:
```
./tracker/eval/calcualte_fpn2mesh_predictions.sh
./tracker/eval/calcualte_fpn2mesh_errors.sh
```
## Mesh Autoencoder
For evaluation of the autoencoder, first perform the predictions then calculate the errors with:
```
./tracker/eval/calcualte_coma_predictions.sh
./tracker/eval/calculate_coma_errors.sh
```

## Tracker
Evaluating the baseline:
```
./tracker/tracking/siam/evaluate_baseline.sh
```
Evaluating Siam Mesh R-CNN
```
./tracker/tracking/siam_mesh/evaluate_siam_mesh.sh
```
Evaluating Siam Mesh R-CNN LC
```
./tracker/tracking/siam_mesh_lc/evaluate_siam_mesh_lc.sh
```


##
Partially re-uses code by 
[Tensorpack](https://github.com/tensorpack/tensorpack) (visited 10.02.2022), 
especially the [Faster R-CNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) (visited 10.02.2022) example, 
to reuse the pre-trained model trained on the COCO dataset. 
The tracker implementation was adapted for the Siam Mesh RCNN from [Siam R-CNN](https://github.com/VisualComputingInstitute/SiamR-CNN) (visited 10.02.2022), by Voigtlaender et al.
The convolutional mesh autoencoder is inspired by [CoMA](https://github.com/anuragranj/coma) (visited 10.02.2022) 
and expanded with a spiral operator as employed by [SpiralNet++](https://github.com/sw-gong/spiralnet_plus) (visited 10.02.2022).


