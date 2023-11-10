**FLOWER CLASSIFICATION**

DATA: [Floral Diversity Dataset](https://www.kaggle.com/datasets/mansi0123/floral-diversity-a-collection-of-beautiful-blooms?select=flower_photos)

- `Data Augmentation`: 3 times
- `Model`: Modified VGG19
- `Train, Val, Test Split`: 65%, 25%, 15%
- `Batch Size`: 64
- `Learning Rate`: 0.001
- `Weight Decay`: 0.0001
- `Optimizer`: Adam
- `Loss Function`: Cross Entropy

I'm using EPOCH 6:

|            | TRAIN   | VAL     | TEST    |
|------------|---------|---------|---------|
| LOSS       | 24.5733 | 21.2500 | 22.0000 |
| ERROR      | 0.1931  | 0.1854  | 0.1798  |
| ACCURACY % | 80.6894 | 81.4613 | 82.0163 |
