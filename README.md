PyTorch DGCNN
=============

About
-----

PyTorch implementation of DGCNN (Deep Graph Convolutional Neural Network). Check https://github.com/muhanzhang/DGCNN for more information.

Installation
------------

This implementation is based on Hanjun Dai's PyTorch version of structure2vec. Please first clone it to this folder by

    git clone https://github.com/Hanjun-Dai/pytorch_structure2vec

Or, alternatively, you can directly unzip the pytorch_structure2vec-master.zip.

Then, under the "pytorch_structure2vec-master/s2vlib/" directory, type

    make -j4

to build the necessary c++ backend.

After that, under the root directory of this repository, type

    ./run_DGCNN.sh

to run DGCNN on dataset DD with default settings.

Or type 

    ./run_DGCNN.sh DATANAME FOLD

to run on another dataset = DATANAME with fold number = FOLD as testing set.

Check "run_DGCNN.sh" for more configuration options.

Reference
---------

If you find the code useful, please cite our paper:

    @inproceedings{zhang2018end,
      title={An End-to-End Deep Learning Architecture for Graph Classification.},
      author={Zhang, Muhan and Cui, Zhicheng and Neumann, Marion and Chen, Yixin},
      booktitle={AAAI},
      year={2018}
    }

Muhan Zhang, muhan@wustl.edu
3/19/2018
