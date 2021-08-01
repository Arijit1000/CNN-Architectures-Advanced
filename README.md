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

AlexNet, proposed by Alex Krizhevsky, uses ReLu(Rectified Linear Unit) for the non-linear part, instead of a Tanh or Sigmoid functions that were the earlier standard for traditional neural networks (like LeNet), ReLu was used in the hidden layers of AlexNet architecture because it trains much faster. This is because the derivative of the sigmoid function becomes very small in the saturating region and therefore the updates applied to the weights almost vanish. This phenomenon is called the vanishing gradient problem. ReLU is represented by this equation f(x) = max(0,x) and is discussed in details in chapter 2.
 
Certain activation functions, like the sigmoid function, squishes a large input space into a small input space between 0 and 1 (-1 to 1 range for Tanh activations). Therefore, a large change in the input of the sigmoid function will cause a small change in the output. Which makes the derivative become very small.
