import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # load the model and weights using the session and vgg path
    tf.saved_model.loader.load(sess,[vgg_tag],vgg_path)
    
    graph = tf.get_default_graph() # Import the graph
    
    # Create layer variables from the given names above
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name) 
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name) 
    lyr3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    lyr4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    lyr7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return w1, keep, lyr3, lyr4, lyr7

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # Create 1x1 convolution layer with kernel size 1 and regularizer:
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out,num_classes,1,padding='SAME',
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    #kernel 4, stride 2, equals upsampling by 2, important: padding the same to match output size
    output = tf.layers.conv2d_transpose(conv_1x1,num_classes,4,2,padding='SAME',
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    #skip connections, make sure the shapes are the same.
    layer4_logit = tf.layers.conv2d(vgg_layer4_out, num_classes,1,padding='SAME',kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    skip = tf.add(output, layer4_logit)
    
    #follow with another transpose upsampling layer, upsample by 2
    output = tf.layers.conv2d_transpose(skip,num_classes,4,2,padding='SAME',
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    #Skip layer again for layer 3 output:
    layer3_logit = tf.layers.conv2d(vgg_layer3_out, num_classes,1,padding='SAME',kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    skip = tf.add(output, layer3_logit)
    
    #follow again with transpose upsampling, upsample by 8
    output = tf.layers.conv2d_transpose(skip,num_classes,16,8,padding='SAME',
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    #return the output layer
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # Reshape the logits and labels from 4D to 2D:
    logits = tf.reshape(nn_last_layer, (-1,num_classes))
    labels = tf.reshape(correct_label, (-1,num_classes))
    
    _entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(_entropy_loss)
    
    return logits, train_op, _entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    
    start_time = time.clock()
    # Init the tf variables:
    sess.run(tf.global_variables_initializer())
    
    epoch_count = 0
    # loop through epochs, get the image and label using the get_batches_fn function    
    for i in range(epochs):
        
        # Debug output
        train_loss = 0
        train_samples = 0
        epoch_count += 1
        print("Running Epoch {}".format(epoch_count))        
        epoch_start_time = time.clock()        
        
        # Setup training template:
        for image,label in get_batches_fn(batch_size):
            train_samples += len(image)
            loss, _ = sess.run([cross_entropy_loss,train_op],feed_dict={input_image:image, correct_label:label, keep_prob:0.8})
            train_loss += loss
        
        # print out training loss
        train_loss /= train_samples
        epoch_end_time = time.clock()
        epoch_time = epoch_end_time - epoch_start_time
        print("Epoch {} ran for {} seconds with a training loss of {}".format(epoch_count,epoch_time,train_loss))
    
    end_time = time.clock()
    tot_time = end_time-start_time
    
    tot_time /= 60
    
    print("Finished all epochs after {} minutes".format(tot_time))                        
    
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    
    # Define epochs batch size and learning rate
    epochs = 25
    batch_size = 1
    lr = 0.0001
    learning_rate = tf.constant(lr)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        shape = [None, image_shape[0], image_shape[1], 3] # setup 4D shape
        # need to define the correct label in the right shape
        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess,vgg_path)
        output_layer = layers(layer3_out,layer4_out,layer7_out,num_classes)
        logits, train_op, cross_entropy_loss = optimize(output_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
