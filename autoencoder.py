import os
from torchvision.datasets import MNIST
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class AutoEncoder():
    """
    Simple implementation of an
    autoencoder using numpy
    """
    def __init__(self, epochs, batch_size, learning_rate,\
                 enc_layers, dec_layers, latent_nuerons,\
                 activation):
        np.random.seed(42)
        self.epochs = epochs
        self.b_size = batch_size
        self.lr = learning_rate
        self.layers = [*enc_layers, latent_nuerons, *dec_layers]
        self.num_layers = len(self.layers)
        self.biases = [np.zeros(layer) for layer in self.layers[1:]]
        self.activations = [np.empty(layer) for layer in self.layers]
        self.act_function = activation
        self.weights = self.init_weights()

    def init_weights(self):
        """
        Initialize weights. 
        Currently only random initialization
        from normal dist is implemented.
        Model weights are lists containing
        an np array for each layer of 
        the model.
        """
        weights = []
        for idx in range(len(self.layers)-1):
            this_layer_neurons = self.layers[idx]
            next_layer_neurons = self.layers[idx+1]
            if self.act_function == 'relu': # uniform weights
                weights.append(np.random.uniform(low=-0.1, high=0.1, \
                        size=(this_layer_neurons, next_layer_neurons)))
            elif self.act_function == 'sigmoid': # He initialization
                sigma = np.sqrt(2/(self.layers[0] + self.layers[-1]))
                weights.append(sigma*np.random.randn(this_layer_neurons, next_layer_neurons))
            else: # default
                weights.append(np.random.randn(this_layer_neurons, next_layer_neurons))
        return weights

    def get_mnist_loader(self, shuffle, train=True, batch_size=None):
        """
        Get the data loader for the 
        mnist dataset.
        """
        MNIST_DIR_NAME = "MNIST"
        DOWNLOAD = False if os.path.exists(MNIST_DIR_NAME) else True

        mnist_trans = transforms.Compose([
            transforms.ToTensor(),
        ])

        mnist_data = MNIST(MNIST_DIR_NAME, download=DOWNLOAD,\
                                transform=mnist_trans, train=train)
        data_loader = DataLoader(mnist_data,
                                batch_size=self.b_size if batch_size is None else batch_size,
                                shuffle=shuffle)
        return data_loader

    def reconstruction_loss(self, inp, out): #, mode='model'
        """
        Calculate the loss caused by 
        the difference between the 
        model's constructed output 
        and the original input.
        Assumes both input and output
        are numpy arrays. 

        If mode=model, just take the 
        difference between outer layers.
        If mode=layer, take the weighted
        sum of the differences between 
        intermediate layers as well.
        """
        assert inp.shape == out.shape
        return out-inp

    def d_recon_loss(self, inp, out): # , mode='model'
        """
        Calculate the derivative
        of the reconstruction loss 
        with respect to model output.
        """
        assert inp.shape == out.shape
        return 1

    def non_lin(self, x):
        """
        Compute the output of a 
        non-linear activation function
        for use in the forward pass 
        of the model.
        """
        if self.act_function == "relu":
            return x * (x>0)
        elif self.act_function == 'sigmoid':
            return np.exp(x) / (1 + np.exp(x))
        else:
            raise NotImplementedError

    def d_nonlin(self, x):
        """
        Compute the derivative of 
        the activation function.
        """
        if self.act_function == "relu":
            return 1. * (x>0)
        elif self.act_function == "sigmoid":
            return x * (1-x)
        else:
            raise NotImplementedError

    def forward(self, x):
        """
        Forward pass of the autoencoder.
        """
        for i, (w,b) in enumerate(zip(self.weights, self.biases)):
            self.activations[i] = x
            x = self.non_lin(x@w + b)
        self.activations[i+1] = x # last set of activations is just model output
        return x

    def backward(self):
        """
        Perform back-propagation
        through the model to update
        its parameters
        """
        out = self.activations[-1]
        inp = self.activations[0]
        deltas = [self.reconstruction_loss(inp, out)]

        for layer in np.arange(len(self.activations) - 2, 0, -1):
            delta = deltas[-1].dot(self.weights[layer].T) * self.d_nonlin(self.activations[layer])
            deltas.append(delta)
        
        deltas = deltas[::-1] # reverse deltas

		# update weights
        for layer in np.arange(0, len(self.weights)):
            self.weights[layer] += -self.lr * self.activations[layer].T.dot(deltas[layer])
            self.biases[layer] += -self.lr * np.mean(deltas[layer], axis=0) # take mean over batches

    def train(self):
        """
        Train the autoencoder :)
        """
        loader = self.get_mnist_loader(shuffle=True, train=True)
        losses = []
        for e in tqdm(range(self.epochs)):
            epoch_loss = 0

            for (images, labels) in loader:
                images = images.cpu().resize(labels.size()[0], 28**2).numpy()
                out = self.forward(images)
                loss = self.reconstruction_loss(images, out)
                self.backward()
                epoch_loss += np.sum(np.abs(loss))
            losses.append(epoch_loss)
        print('done training: ', losses)
        return losses

    def show_loss_curve(self, losses):
        """
        Plot the epoch losses
        using matplotlib
        """
        plt.plot(range(1, len(losses)+1), losses)
        plt.title('Loss curve after training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def visualize(self, train=False, num_imgs=5):
        """
        Using the first batch 
        out of the train or test 
        dataloader, visualize
        some of the results. This
        function will show an 
        image and next to it 
        the autoencoder's 
        recreation of that
        same image.
        """
        loader = self.get_mnist_loader(shuffle=False, train=train, batch_size=num_imgs)
        for images, labels in loader:
            images = images.cpu().resize(labels.size()[0], 28**2).numpy()
            output = self.forward(images)
            for i, (img, out) in enumerate(zip(images, output)):
                plt.imshow(np.resize(img, (28,28)), interpolation='nearest')
                plt.title(f'Input {i+1}')
                plt.gray()
                plt.show()
                plt.imshow(np.resize(out, (28,28)), interpolation='nearest')
                plt.title(f'Output {i+1}')
                plt.gray()
                plt.show()
            break
