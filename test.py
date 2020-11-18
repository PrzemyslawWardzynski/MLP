from MultiLayerPerceptron import *
import mnist_loader
import matplotlib.pyplot as plt
import numpy as np
training, validation, test = mnist_loader.load_data()
offset = 5000
t = training[0][0:offset],training[1][0:offset]
offsetv = 500
v = validation[0][0:offsetv],validation[1][0:offsetv]

_,a = np.unique(training[1],return_counts=True)
b = np.arange(0,10,1)
plt.bar(b,a)
plt.title("Częstotliwość występowania klas w zbiorze treningowym")
plt.xlabel('Klasa')
plt.ylabel('Częstotliwość występowania w zbiorze treningowym')
plt.savefig('train_frequency.png')
plt.show()
print(a)

hidden_layer_size = 397
weight_range = (-0.4,0.4)
epochs = 100
learn_step = 0.05
batch_size = 64
activation_function = softplus_function
derivative_function = sigmoid_function
patience = 40
m = MultiLayerPerceptron([784,hidden_layer_size,10],weight_range)
_,epochs_list,accuracy_list,max_accuracy = m.train(t,v,epochs,learn_step,batch_size,activation_function,derivative_function,patience)
plt.title("Uczenie na pełnym zbiorze trenującym")
plt.plot(epochs_list,accuracy_list)
plt.xlabel('epoki')
plt.ylabel('skuteczność')
plt.savefig('best_parameters_learning.png')
print(max_accuracy)
plt.show()

reps = 5
#TEST HIDDEN LAYER SIZE

reps = 5
hidden_layer_sizes = [10,20,50,100,200,397]
weight_range = (-1,1)
epochs = 250
learn_step = 0.1
batch_size = 1667
activation_function = sigmoid_function
derivative_function = sigmoid_derivative
patience = 20

accuracies = []
x_list = []
y_list = []
for l_size in hidden_layer_sizes:
    
    
    x = []
    
    for rep in range(0,reps):

        m = MultiLayerPerceptron([784,l_size,10],weight_range)
        _,x,y,max_accuracy = m.train(t,v,epochs,learn_step,batch_size,activation_function,derivative_function,20)
        accuracies.append(max_accuracy)
        avg_x = [element / reps for element in x]
        x_list.append(avg_x)
        avg_y = [element / reps for element in y]
        y_list.append(avg_y)
    
    plt.title('Skuteczność sieci w kolejnych epokach, l. neuronów w warstwie ukrytej = {0}'.format(l_size))
    plt.xlabel('epoki')
    plt.ylabel('skuteczność')
    plt.plot(x, y, '-',)
    plt.savefig('hidden_layer_learning_{0}.png'.format(l_size))
    plt.cla()

plt.title('Wpływ liczby neuronów w warstwie ukrytej na jakość uczenia')
plt.xlabel('l.neuronów w warstwie ukrytej')
plt.ylabel('skuteczność')

xticks = np.arange(len(hidden_layer_sizes))
plt.bar(xticks, accuracies, color ='maroon',  
        width = 0.8)
plt.xticks(xticks,hidden_layer_sizes)
plt.savefig('hidden_layer_accuracy')
plt.cla()
    
plt.title('Skuteczność sieci w kolejnych epokach dla liczby N neuronów warstwy ukrytej')
plt.xlabel('epoki')
plt.ylabel('skuteczność')
for i in range(0,len(hidden_layer_sizes)):
    size = hidden_layer_sizes[i]
    plt.plot(x_list[i],y_list[i],label='N = ' + str(size))
plt.legend()
plt.savefig('hidden_layer_learning_summary.png')
plt.cla()



#TEST BATCH SIZE

hidden_layer_size = 100
mini_batch_sizes = [1,2,4,8,16,32,64,128,256,512]
weight_range = (-1,1)
epochs = 250
learn_step = 0.1

activation_function = sigmoid_function
derivative_function = sigmoid_derivative
patience = 20

accuracies = []
x_list = []
y_list = []
for mini_batch_size in mini_batch_sizes:
    
    
    x = []
    
    for rep in range(0,reps):

        m = MultiLayerPerceptron([784,hidden_layer_size,10],weight_range)
        _,x,y,max_accuracy = m.train(t,v,epochs,learn_step,mini_batch_size,activation_function,derivative_function,20)
        accuracies.append(max_accuracy)
        avg_x = [element / reps for element in x]
        x_list.append(avg_x)
        avg_y = [element / reps for element in y]
        y_list.append(avg_y)
    
    plt.title('Skuteczność sieci w kolejnych epokach, rozmiar batcha = {0}'.format(mini_batch_size))
    plt.xlabel('epoki')
    plt.ylabel('skuteczność')
    plt.plot(x, y, '-',)
    plt.savefig('mini_batch_learning_{0}.png'.format(mini_batch_size))
    plt.cla()

plt.title('Wpływ rozmiaru batcha na jakość uczenia')
plt.xlabel('Rozmiar batcha')
plt.ylabel('skuteczność')

xticks = np.arange(len(mini_batch_sizes))
plt.bar(xticks, accuracies, color ='maroon',  
        width = 0.8)
plt.xticks(xticks,mini_batch_sizes)
plt.savefig('mini_batch_accuracy.png')
plt.cla()
    
plt.title('Skuteczność sieci w kolejnych epokach dla liczby N rozmiaru batcha')
plt.xlabel('epoki')
plt.ylabel('skuteczność')
for i in range(0,len(mini_batch_sizes)):
    size = mini_batch_sizes [i]
    plt.plot(x_list[i],y_list[i],label='N = ' + str(size))
plt.legend()
plt.savefig('mini_batch_summary.png')
plt.cla()

#TEST WEIGHT RANGE

hidden_layer_size = 100
mini_batch_size = 64
weight_ranges = [(-1,1),(-0.8,0.8),(-0.6,0.6),(-0.4,0.4),(-0.2,0.2),(-0.1,0.1)]
epochs = 250
learn_step = 0.1

activation_function = sigmoid_function
derivative_function = sigmoid_derivative
patience = 20

accuracies = []
x_list = []
y_list = []
for weight_range in weight_ranges:
    
    
    x = []
    
    for rep in range(0,reps):

        m = MultiLayerPerceptron([784,hidden_layer_size,10],weight_range)
        _,x,y,max_accuracy = m.train(t,v,epochs,learn_step,mini_batch_size,activation_function,derivative_function,20)
        accuracies.append(max_accuracy)
        avg_x = [element / reps for element in x]
        x_list.append(avg_x)
        avg_y = [element / reps for element in y]
        y_list.append(avg_y)
    
    plt.title('Skuteczność sieci w kolejnych epokach, przedział inicjalizacji wartości wag  = {0}'.format(weight_range))
    plt.xlabel('epoki')
    plt.ylabel('skuteczność')
    plt.plot(x, y, '-',)
    plt.savefig('weight_range_learning_{0}.png'.format(weight_range))
    plt.cla()

plt.title('Wpływ inicjalizacji wartości wag na jakość uczenia')
plt.xlabel('Przedział inicjalizacji wag')
plt.ylabel('skuteczność')

xticks = np.arange(len(weight_ranges))
plt.bar(xticks, accuracies, color ='maroon',  
        width = 0.8)
plt.xticks(xticks,weight_ranges)
plt.savefig('weight_range_accuracy.png')
plt.cla()
    
plt.title('Skuteczność sieci w kolejnych epokach dla przedziału inicjalizacji wag')
plt.xlabel('epoki')
plt.ylabel('skuteczność')
for i in range(0,len(weight_ranges)):
    size = weight_ranges[i]
    plt.plot(x_list[i],y_list[i],label='N = ' + str(size))
plt.legend()
plt.savefig('weight_range_summary.png')
plt.cla()

"""
"""
#TEST LEARN STEP

hidden_layer_size = 100
mini_batch_size = 64
weight_range = (-0.4,0.4)
epochs = 250
learn_step_values = [0.005,0.01,0.05,0.1,0.2,0.5,1]

activation_function = sigmoid_function
derivative_function = sigmoid_derivative
patience = 20

accuracies = []
x_list = []
y_list = []
for learn_step in learn_step_values:
    
    
    x = []
    
    for rep in range(0,reps):

        m = MultiLayerPerceptron([784,hidden_layer_size,10],weight_range)
        _,x,y,max_accuracy = m.train(t,v,epochs,learn_step,mini_batch_size,activation_function,derivative_function,20)
        accuracies.append(max_accuracy)
        avg_x = [element / reps for element in x]
        x_list.append(avg_x)
        avg_y = [element / reps for element in y]
        y_list.append(avg_y)
    
    plt.title('Skuteczność sieci w kolejnych epokach, współczynnik uczenia  = {0}'.format(learn_step))
    plt.xlabel('epoki')
    plt.ylabel('skuteczność')
    plt.plot(x, y, '-',)
    plt.savefig('learn_step_learning_{0}.png'.format(learn_step))
    plt.cla()

plt.title('Wpływ współczynnika uczenia na jakość uczenia')
plt.xlabel('współczynnik uczenia')
plt.ylabel('skuteczność')

xticks = np.arange(len(learn_step_values))
plt.bar(xticks, accuracies, color ='maroon',  
        width = 0.8)
plt.xticks(xticks,learn_step_values)
plt.savefig('learn_step_accuracy.png')
plt.cla()
    
plt.title('Skuteczność sieci w kolejnych epokach dla współczynników uczenia N')
plt.xlabel('epoki')
plt.ylabel('skuteczność')
for i in range(0,len(learn_step_values)):
    size = learn_step_values[i]
    plt.plot(x_list[i],y_list[i],label='N = ' + str(size))
plt.legend()
plt.savefig('learn_step_summary.png')
plt.cla()
"""
#TEST SIGM/RELU(SOFTPLUS) ACTIVATION FUNCTION
"""
hidden_layer_size = 100
mini_batch_size = 64
weight_range = (-0.4,0.4)
epochs = 250
learn_step= 0.05

activation_functions = [sigmoid_function,softplus_function]
derivative_functions = [sigmoid_derivative,sigmoid_function]
patience = 20

accuracies = []
x_list = []
y_list = []
for i in range(len(activation_functions)):
    
    activation_function = activation_functions[i]
    derivative_function = derivative_functions[i]
    
    x = []
    
    for rep in range(0,reps):

        m = MultiLayerPerceptron([784,hidden_layer_size,10],weight_range)
        _,x,y,max_accuracy = m.train(t,v,epochs,learn_step,mini_batch_size,activation_function,derivative_function,20)
        accuracies.append(max_accuracy)
        avg_x = [element / reps for element in x]
        x_list.append(avg_x)
        avg_y = [element / reps for element in y]
        y_list.append(avg_y)
    
    if i == 0:
        plt.title('Skuteczność sieci w kolejnych epokach, f.aktyw. sigmoidalna')
    else:
        plt.title('Skuteczność sieci w kolejnych epokach, f.aktyw. Softplus')
    plt.xlabel('epoki')
    plt.ylabel('skuteczność')
    plt.plot(x, y, '-',)
    plt.savefig('activation_function_learning_{0}.png'.format(i))
    plt.cla()

plt.title('Wpływ funkcji aktywacji na jakość uczenia')
plt.xlabel('Funkcja aktywacji')
plt.ylabel('skuteczność')

xticks = np.arange(len(activation_functions))
plt.bar(xticks, accuracies, color ='maroon',  
        width = 0.8)
plt.xticks(xticks,['sigmoidalna','Softplus'])
plt.savefig('activation_function_accuracy.png')
plt.cla()
    
plt.title('Skuteczność sieci w kolejnych epokach dla wybranych funkcji aktywacji')
plt.xlabel('epoki')
plt.ylabel('skuteczność')
plt.plot(x_list[0],y_list[0],label='sigmoidalna')
plt.plot(x_list[1],y_list[1],label='Softplus')
plt.legend()
plt.savefig('activation_function_summary.png')
plt.cla()
