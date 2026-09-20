import numpy as np
import graph
import Data_extraction as data
from matplotlib import pyplot as plt

#baisic ffn with cross entorpy loss and leaky relu activation function as an example of how to use my autograd system, lets you see the loss drop


class FFN():
    def __init__(self,sizes,alpha):
        self.layers = []

        self.input_size = sizes[0]
        for i in range(len(sizes)-1):
            self.layers.append(Layer(sizes[i],sizes[i+1],alpha,(i == len(sizes)-2)))

    def forwards(self,input):

        curr_output = graph.root(input,False)

        for layer in self.layers:
            curr_output = layer.forwards(curr_output)
        self.output = curr_output
        self.logits = self.layers[-1].z
        return self.output.value()

    def backwards(self,lr,expected_result):#do the backwards pass(handled by graph) and clean up 

        self.elementwise_loss = graph.SoftmaxCrossEntropy((self.logits,),(-1,expected_result))

        #make the divisor into a numpy array so that the operation works, all operations are built for numpy arrays
        self.loss = graph.Summation((self.elementwise_loss,),((),False)) * np.array(1/(np.shape(self.elementwise_loss.value())[0]))

        graph.backwards_pass(graph.node.levels,loss_nodes=(self.loss,))
        graph.SGD(graph.node.levels,lr)

        loss = self.loss.value()

        graph.cleanup(graph.node.levels)
        return loss

    def train_batch(self,input,expected_output,lr):
        output = self.forwards(input)
        loss = self.backwards(lr,expected_output)
        return (loss,output)



class Layer():
    def __init__(self,input_size,output_size,alpha,is_output):
        self.is_output = is_output
        self.alpha = alpha

        if is_output:
            std = np.sqrt(1.0 / input_size)   # Xavier/Glorot for softmax
        else:
            std = np.sqrt(2.0 / ((1 + alpha**2) * input_size))
        matrix = np.random.randn(input_size, output_size) * std
        
        baises = np.zeros(shape=(1,output_size))
        self.wheigts = graph.root(matrix,True)
        self.baises = graph.root(baises,True)

    def forwards(self,inputs):
        self.z = (inputs @ self.wheigts)
        self.z +=self.baises

        if self.is_output:
            self.a = graph.Softmax((self.z,),-1)
        else:
            self.a = graph.LeakyRelu((self.z,),self.alpha)
       
        return self.a



def train(model,training_images,training_lables,iterations,batch_size,lr):
    cost_data = []
    for i in range(iterations):
        #take random samples
        random_indicies = np.random.choice(np.shape(training_images)[0],size = batch_size, replace=False)
        inputs = training_images[random_indicies]
        expected_output = training_lables[random_indicies]

        #put the batch throught the neural network
        cost,output = model.train_batch(inputs,expected_output,lr)
        print(f'predicted:{np.argmax(output[0])}, real: {np.argmax(expected_output[0])},{np.argmax(output[0]) == np.argmax(expected_output[0])} conf {np.max(output[0])}')
        cost_data.append(cost)

    cost_data = np.array(cost_data)
    return cost_data



sizes = [784,258,150,10]
myffn = FFN(sizes,0.01)


training_ims = data.extract_data_images('emnist-digits-train-images-idx3-ubyte')
training_lables = data.extract_data_labels('emnist-digits-train-labels-idx1-ubyte')

iterations = 200

cost_data = train(myffn,training_ims,training_lables,iterations,100,0.001)


plt.plot(np.arange(iterations),cost_data)
plt.show()