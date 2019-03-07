# Barebones neural network
## Special thanks
This network was built with the help of Michael Nielsen's book [Neural Networks and Deep learning](http://neuralnetworksanddeeplearning.com/chap1.html) and his [network](https://github.com/mnielsen/neural-networks-and-deep-learning) was used as an reference.
The code, as its inspiration tries to focus on readability and ease of modification. Different types of backward propagation (GBD with momentum etc...) can be addded with almost no modification (and they will be, soon), other cost functions are also easy to implement (coming up!)
Huge thanks to Michael Nielsen for providing such great resources online.
## Using the network
The network can be run off-the-shelf.
Steps to run:
1. Import the network
2. Create an instance of network, giving an array representing the amount of neurons in each layer [x0,x1,x2..xn].
NB. x0 and xn have to correspond to the height of the input and output vector respectively
3. Train! Run Network.SGD(training_data, epochs, mini_batch_size, eta, test_data*)
The parameters are self explanotary, but a couple of notes:
training_data and optional test_data should be a tuple (x,y) x and y being arrays of input and output vectors respectively.
Network can run batch gradient descent and stochastic gradient descent if mini_batch_size is set to length of the training data or 1
NB. Network.save saves the weights and biases of the network in a .pickle file
## Future plans
The network will gain new functionality in the following weeks,(Momentum gradient descent, new cost functions...) and will be used to construct an chess-playing neural net.
## Dependencies
The network runs in python3, and uses numpy, random and _pickle
