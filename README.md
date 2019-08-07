# Facial Expression Recognition

## Demo
The model performs really well in the real world and it has an accuracy of 75% on the test data.

| ![demo.gif](doc/demo.gif) | 
|:--:| 
| *The model has 5 class. (Happy,Sad,Surprise,Angry,Neutral)* |

## Datasets
The Datasets which I have used in this project are [AffectNet]('http://mohammadmahoor.com/affectnet') and [FER+](https://github.com/microsoft/FERPlus) (which is the same as fer2013 Kaggel but with better labels)

The combination of these dataset will have about 300,000 images for the 5 class.

| Class         | Number of Images |
| ------------- |:----------------:|
| Happy         | 134,626          |
| Neutral       | 86,518           |
| Sad           | 29,886           |
| Angry         | 28,168           |
| Surprise      | 18,608           |

## Haar-Cascade

[Haar cascade classifiers](https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html) first were proposed by Paul Viola and Michael Jones and uses Haar kernels for extracing features and it uses multiple classifiers one after an other. The classifiers get more complex as data move forward. Each classifier will specify whether the image is maybe from the desired class or it is definitly not in the desired class and if it is maybe from the desired class will pass image forward to the next classifier.

For training these classifiers,first you need to collect images from desired class (positive data in our case face images)  and everything else (negative data) especially from the environment which you want to test your model like chairs, tables, etc.
(for better result its better to make the images grayscale). Also you need to make a description file for postive and negetive samples. 
* For positives you need to write in the following format:
\[filename] \[number of object annotations] \[coordinates of the objects bounding rectangles (x, y, width, height)]
for example: img/img1.jpg  1  140 100 45 45
* For negative images you only need to write the file name

After collecting data and creating the two description files, you have to install opencv and its dependencies. Then using following commands you can start training your model:

First you need to create a vector file from the positive image with the following command:
opencv_createsamples -info \[name of description file] -num \[number of positive samples] -w \[width of the output] -h \[height of the output] -vec \[name of the vector file]
for example:
```opencv_createsamples -info info/info.lst -num 9000 -w 20 -h 20 -vec positives.vec```

Then after creating the vector file you can start the actuall training with the following command:
```opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 7000 -numNeg 3500 -numStages 10 -w 20 -h 20```

For more information you can reffer to [opencv website](https://docs.opencv.org/3.4.3/dc/d88/tutorial_traincascade.html).

## CNN

The cnn was implemented in Keras with the following architecture:

```
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_20 (Conv2D)           (None, 46, 46, 64)        640       
_________________________________________________________________
activation_35 (Activation)   (None, 46, 46, 64)        0         
_________________________________________________________________
batch_normalization_30 (Batc (None, 46, 46, 64)        256       
_________________________________________________________________
max_pooling2d_20 (MaxPooling (None, 23, 23, 64)        0         
_________________________________________________________________
conv2d_21 (Conv2D)           (None, 21, 21, 128)       73856     
_________________________________________________________________
dropout_10 (Dropout)         (None, 21, 21, 128)       0         
_________________________________________________________________
activation_36 (Activation)   (None, 21, 21, 128)       0         
_________________________________________________________________
batch_normalization_31 (Batc (None, 21, 21, 128)       512       
_________________________________________________________________
max_pooling2d_21 (MaxPooling (None, 10, 10, 128)       0         
_________________________________________________________________
conv2d_22 (Conv2D)           (None, 8, 8, 256)         295168    
_________________________________________________________________
activation_37 (Activation)   (None, 8, 8, 256)         0         
_________________________________________________________________
batch_normalization_32 (Batc (None, 8, 8, 256)         1024      
_________________________________________________________________
max_pooling2d_22 (MaxPooling (None, 4, 4, 256)         0         
_________________________________________________________________
conv2d_23 (Conv2D)           (None, 2, 2, 256)         590080    
_________________________________________________________________
activation_38 (Activation)   (None, 2, 2, 256)         0         
_________________________________________________________________
batch_normalization_33 (Batc (None, 2, 2, 256)         1024      
_________________________________________________________________
max_pooling2d_23 (MaxPooling (None, 1, 1, 256)         0         
_________________________________________________________________
flatten_5 (Flatten)          (None, 256)               0         
_________________________________________________________________
dense_15 (Dense)             (None, 128)               32896     
_________________________________________________________________
activation_39 (Activation)   (None, 128)               0         
_________________________________________________________________
batch_normalization_34 (Batc (None, 128)               512       
_________________________________________________________________
dropout_11 (Dropout)         (None, 128)               0         
_________________________________________________________________
dense_16 (Dense)             (None, 64)                8256      
_________________________________________________________________
activation_40 (Activation)   (None, 64)                0         
_________________________________________________________________
batch_normalization_35 (Batc (None, 64)                256       
_________________________________________________________________
dense_17 (Dense)             (None, 5)                 325       
_________________________________________________________________
activation_41 (Activation)   (None, 5)                 0         
=================================================================
Total params: 1,004,805
Trainable params: 1,003,013
Non-trainable params: 1,792
```

## References
[Rapid Object Detection usinga Boosted Cascade of Simple Features](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)
[AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild](https://arxiv.org/abs/1708.03985)

[Kaggel](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)




