# Semantic Segmentation
### Introduction
In this project, a Fully Convolutional Network (FCN) was used to label the pixels of a road in images from the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php).

This project only considered the roadway ahead as a type of feature to learn to detect. The data set does contain an option for including 'other' roads as an additional feature but that has not been implemented here. There were no augmentation techniques used for the training.

### Model and Setup
##### Packages
Packages used:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
 
 ##### Framework - Computing on AWS
 An EC2 AWS spot instance with the following specifications was setup to train the models:
 
 * Region: US-West Oregon
 * Instance Type:g3.4xlarge
 * GPUs: 1
 * GPU Memory: 8 GB
 * vCPUs: 16
 * Main Memory: 122 GB
 * EBS Bandwidth: 3.5Gbps
 
 AWS Security key was configured to accept incoming SSH from my local IP address. The instance was also prevented from deleting the Volume if the instance was interrupted.
 
 The AWS instance was created with an AMI template from Udacity: udacity-carnd-advanced-deep-learning - ami-3e6c7547
 This allowed for an ubuntu environment with the required anaconda, python, and cuda drivers setup already.
 
 Windows Subsystem for Linux was used to SSH into the AWS instance, and training could begin.
 
 #### TensorFlow Model
 A Fully Convolutional Network was setup by pre-loading the Encoder portion using the VGG neural net.
 
 The FCN Architecture is shown in the schematic:
 ![FCN](fcn_2.png)
 
 The Decoder portion of the FCN required the addition of a 1x1 convolutional layer to the end of each of the Encoder output layer that were used in the Decoder. Layers 3,4 and 7 were downsampled through the central 1x1 convolutional layer which allows for teaching the network what a roadway looks like, but that information path alone does not complete the FCN model. By using Skip (tf.add) for layers 3, and 4, the spatial relation information of the roadway is preserved.
 
 An Adam Optimizer was used with a softmax cross-entropy loss using logits:
 `_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))`
 
 The FCN was setup with the following hyper-parameters:
 
 *  Epochs = 25
 *  batch size = 1
 *  learning rate = 0.0001
 
 The model took approximately 17 minutes to train 25 epochs.
 
 ### Results
 Examples of the Semantic Segmentation:
 ![runexample1](results1.png)
 ![runexample2](results2.png)
 