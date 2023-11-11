# **Flower Classification**
## Dataset
The data is obtained from [Kaggle](https://www.kaggle.com/datasets/mansi0123/floral-diversity-a-collection-of-beautiful-blooms?select=flower_photos). There are 5 different class (Daisy, Dandelion, Roses, Sunflowers and Tulips). Additionally, there are 3671 images with each class having around 600 to 900 pictures.


## Model
A pretrained **VGG 19** model is used for this project. The classifier layers were replaced with 2 fully connected layers. The model is then trained with the downloaded data. The data was also augmented with 3 different transformation. These transformations use Color Jitter, Gaussian Blur, Rotation and Vertical Flip. 

Training Hyperparameters:
- `Model`: Modified VGG19
- `Train, Val, Test Split`: 65%, 25%, 15%
- `Batch Size`: 64
- `Learning Rate`: 0.001
- `Weight Decay`: 0.0001
- `Optimizer`: Adam
- `Loss Function`: Cross Entropy

## Results:

|      Epoch 6      | TRAIN   | VAL     | TEST    |
|------------|---------|---------|---------|
| LOSS       | 24.5733 | 21.2500 | 22.0000 |
| ERROR      | 0.1931  | 0.1854  | 0.1798  |
| ACCURACY % | 80.6894 | 81.4613 | 82.0163 |
