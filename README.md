# CNN-Architectures-Advanced

AlexNet architecture
You’ve seen a version of the AlexNet architecture in the project at the end of chapter 3. The architecture is pretty straightforward. It consists of:
LeNet-5 is a small neural network with today’s standards. It has 61,706 parameters compared to millions of parameters in more modern networks as you will see later in this chapter in more modern architectures.

## Set up the learning hyperparameters
The authors used a scheduled decay learning where the value of the learning rate was decreasing using the following schedule: 0.0005 for the first two epochs, 0.0002 for the next three epochs, 0.00005 for the next four, then 0.00001 thereafter. In their paper, the authors trained their network for 20 epochs.

Let’s build a lr_schedule function with the above schedule. The method will take an integer epoch number as an argument and returns the learning rate (lr).

AlexNet is consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax. You can represent the AlexNet architecture in text as follows:
INPUT IMAGE  => CONV1 => POOL2 => CONV3 => POOL4 => CONV5 => CONV6 => CONV7 => POOL8 => FC9 => FC10 => SOFTMAX7

## Novel features of AlexNet
Before AlexNet, deep learning was starting to gain traction in speech recognition and a few other areas. But AlexNet was the milestone that convinced a lot of the computer vision community to take a serious look at deep learning and demonstrate that deep learning really works in computer vision. Compared to previous CNNs (like LeNet), AlexNet presented some novel features that were not used in previous architectures. You are already familiar of all of them from the previous chapters in this book so it should be quick for us to go through them here.

## ReLU activation function:

AlexNet, proposed by Alex Krizhevsky, uses ReLu(Rectified Linear Unit) for the non-linear part, instead of a Tanh or Sigmoid functions that were the earlier standard for traditional neural networks (like LeNet), ReLu was used in the hidden layers of AlexNet architecture because it trains much faster. This is because the derivative of the sigmoid function becomes very small in the saturating region and therefore the updates applied to the weights almost vanish. This phenomenon is called the vanishing gradient problem. ReLU is represented by this equation f(x) = max(0,x).
 
Certain activation functions, like the sigmoid function, squishes a large input space into a small input space between 0 and 1 (-1 to 1 range for Tanh activations). Therefore, a large change in the input of the sigmoid function will cause a small change in the output. Which makes the derivative become very small.

## Dropout layer:

as explained in chapter 3, dropout layers are used to avoid the neural network overfitting. The neurons which are “dropped out” do not contribute to the forward pass and do not participate in backpropagation. This means that every time an input is presented, the neural network samples a different architecture, but all these architectures share the same weights. This technique reduces complex co-adaptations of neurons, since a neuron cannot rely on the presence of particular other neurons. It is, therefore, forced to learn more robust features that are useful in conjunction with many different random subsets of the other neurons. The authors used dropout with a probability = 0.5 in the two fully-connected layers.

## Data augmentation:

one popular and very effective approach to avoid overfitting is to artificially enlarge the dataset using label-preserving transformations. This happens by generating new instances of the training images with some transformations like image rotation, flipping, scaling, and many more.

## Local response normalization:

in AlexNet, local response normalization is used. It is different from the batch normalization technique (explained in chapter 4). Normalization helps to speed up the convergence. Nowadays, batch normalization (BN) is used instead of using local response normalization and we will be using BN in our implementation.

## Weight regularization:

the authors used a weight decay of 0.0005. Weight decay is another term for the L2 regularization technique explained in chapter 4. It is an approach to reduce the overfitting of a deep learning neural network models on the training data to allow it to generalize better on new data.
model.add(Conv2D(32, (3,3), kernel_regularizer=l2(λ)))

The lambda value is the weight decay hyperparameter that you can tune. If you still see overfitting size, increase the lambda value to reduce overfitting. In this case, the authors found that a small decay value of 0.0005 was good enough for the model to learn.

## Training on multiple GPUs:

GTX 580 GPU that has only 3GB of memory is used here. It was state-of-the-art at the time but not large enough to train the 1.2 million training examples in their dataset. Therefore they developed a complicated way to spread their network across two GPUs. The basic idea was that, a lot of these layers were split across two different GPUs and there was a thoughtful way for when the two GPUs would communicate with each other. 

##  AlexNet implementation in Keras
The network contains eight weight layers: the first five are convolutional and the remaining three are fully-connected. The output of the last fully-connected layer is fed to a 1000-way softmax which produces a distribution over the 1000 class labels. AlexNet input starts with 227x227x3 images. 
