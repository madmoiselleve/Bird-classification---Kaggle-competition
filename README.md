## Bird-classification - Kaggle-competition
### Classification challenge on a subset of the Caltech-UCSD Birds-200- 2011 bird dataset with pre-trained ResNext-101- 32x8d model.


#### Requirements
1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```


#### Dataset
We will be using a dataset containing 200 different classes of birds adapted from the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
After a quick analysis of the provided dataset, the training set only contains 1087 images with a low informative value. This motivates a data augmentation, and a cropping of the background as it does not appear as a discriminating factor. We will then use a pretrained model to achieve classification. Code is provided on a Kaggle notebook, and we use Pytorch as a deep learning framework.

#### Model
- We use a ResNext-101-32x8d model \cite{Resnext}, pretrained on ImageNet . This model architecture is inspired by ResNet architecture (four main stages of convolution and a fully connected layer), but the Blocks itselves differs. ResNeXt block  uses a "split-transform-merge" strategy, which means it aggregates a list of transformations. I also exposes a new dimension, cardinality (size of set of transformations). 
After unfreezing the first 3 blocks and the fc, adding a final classification layer, we train the model for 16 epochs on the concatenated dataset, with an SGD optimiser. Finally, as recquired, the loss function will be Cross Entropy.

- Several other pretrained models were also tested, with poorer performances. Another approach to improve the model was to perform feature extraction, which consists in only updating the final layer weights. We tried then to progressively unfreeze each layer (finetuning) and compare performances. We finally decided to unfreeze last layers from Layer 3, as we observed that the first layers of a CNN contain generic features that are not specific to a particular task. 

- By default the images are loaded and resized to 64x64 pixels and normalized to zero-mean and standard deviation of 1. 

#### Final Score

This algorithm reaches a 86% accuracy on the whole testing dataset. Several ways to improve this performance would be to play with the optimizer, or fine-tuning other pretrained models like VisualTransformers, or RegNet.

