import random
import tensorflow as tf
import math
import time

from tensorflow.keras import datasets, layers, models, regularizers
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from fxpmath import Fxp
import pickle
from datetime import datetime
import gc
import os
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt 
import matplotlib as mpl
from scipy import optimize 
import seaborn as sns
import sys




#----------------------------USER DEFINED
simulations=1          #Nº of simulations. Set >1 if want multiple cuantization results.
Max_steps=100          #Steps of Simulated Annealing convergence algorithm.

interval = (1, 8)     #Search range when simulating the quantification of the fractional part of the parameters.
max_degradation=5      #Reference based on maximum network accuracy operating in float32 format

    #---------------------------Convergence guidance hyperparameters
    #Cost function=gamma*((lower_bound-actual_acc)**2) + beta*avg_bits -alpha*lower_bound
alpha=0
beta=5
gamma=1
    #---------------------------/Convergence guidance hyperparameters
#----------------------------/USER DEFINED





final_acc_sim=[]
final_avg_sim=[]

try:
    os.mkdir(f"./{simulations}_sims_max_steps_{Max_steps}")
except:
    print("It is not possible to create a new folder to store the simulation results. Stopping the execution")

for n_iter in range(simulations):
#-------GLOBAL VARS
    last_i=-1
    new_cost=-1
    weights_cost=[]
    time_stamp=time.time()
    #-------------------------------Model_data_parameters. MODIFY WITH THE TOPOLOGY OF INTEREST
    num_classes = 10
    input_shape = (28, 28, 1)

    tf.keras.backend.clear_session()
    model = models.Sequential()
    model.add(layers.Conv2D(16, (5, 5),padding="same", activation='relu', input_shape=(32, 32, 3),kernel_regularizer=regularizers.L2(l2=0.0001),)) #The L2 regularization penalty is computed as: loss = l2 * reduce_sum(square(x))
    model.add(layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(layers.Conv2D(20, (5, 5),strides=(1,1),padding="same", activation='relu',kernel_regularizer=regularizers.L2(l2=0.0001) ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(layers.Conv2D(20, (5, 5),strides=(1,1),padding="same", activation='relu',kernel_regularizer=regularizers.L2(l2=0.0001) ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax',kernel_regularizer=regularizers.L2(l2=0.0001)))

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images=(train_images/255.0).astype("float32")
    test_images=(test_images/255.0).astype("float32")

    names=['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']


    try:
        model.load_weights("./weights.h5")
    except:
        print("Problem loading the weights of the model. Did you located it in parent folder? Remember, the name must be 'weights' in h5 format")
        break
    #-------------------------------/Model_data_parameters. MODIFY WITH THE TOPOLOGY OF INTEREST

    w_dict = {}
    for layer in model.layers:
        w_dict[layer.name] = model.get_layer(layer.name).get_weights()

    #--------------------------------------------------------------------------------------------
    sns.set(context="talk", style="darkgrid", palette="hls", font="sans-serif", font_scale=1.05)

    FIGSIZE = (19, 8)  #: Figure size, in inches!
    mpl.rcParams['figure.figsize'] = FIGSIZE

    factor=10 #Neccesary if the cost function involves the square of the difference
    test_acc = model.evaluate(train_images,  train_labels, verbose=0)[1]
    lower_bound=(test_acc-(max_degradation/100))*factor
    n_int=1

    layers_of_interest=[]

    for layer in model.layers:
        if (len(layer.get_weights())>0):
            layers_of_interest.append(layer)
    def f(x,alpha,beta,gamma):
        """ Function to minimize."""
        fxp_by_layer=[]
        for i in range(len(x)):
            fxp_by_layer.append(Fxp(None, signed=True, n_int=n_int, n_frac=x[i]))
        for i, layer in enumerate(layers_of_interest):
            model.get_layer(layer.name).set_weights([Fxp(w_dict[layer.name][0], like=fxp_by_layer[i] ),Fxp(w_dict[layer.name][1], like=fxp_by_layer[i])])

        actual_acc= model.evaluate(train_images,  train_labels, verbose=0)[1]*factor
        a,b=interval
        x_array=np.array([sum(x)/len(layers_of_interest),max(a,b)])

        avg_bits=preprocessing.normalize([x_array])[0][0]
        cost= gamma*((lower_bound-actual_acc)**2) + beta*avg_bits -alpha*lower_bound
        weights_cost.append([gamma*((lower_bound-actual_acc)**2)/cost,beta*avg_bits/cost,alpha*lower_bound/cost])

        return cost

    def clip(x,i):#OK
        """ Force x to be in the interval."""
        a, b = interval
        x[i]=int(max(min(x[i], b), a))
        return x

    def random_start():#OK
        """ Random point in the interval."""
        a, b = interval
        start=[]
        for i in range(0,len(layers_of_interest)):
            start.append(int(round(a + (b - a) * rn.random_sample())))

        return start

    def cost_function(x,alpha,beta,gamma):
        """ Cost of x = f(x)."""
        return f(x,alpha,beta,gamma)

    def random_neighbour(x, T,cost,new_cost):
        """Move a little bit x, from the left or the right."""
        amplitude = int(math.ceil((max(interval) - min(interval))* 0.5 * T))

        if(cost==new_cost or new_cost>cost):
            i=random.randint(0,len(layers_of_interest)-1)
        else:
            i=last_i
    
        delta = amplitude * random.randrange(-1,2,2)
        x[i]=x[i]+delta
        return clip(x,i),i

    def acceptance_probability(cost, new_cost, temperature):
        if new_cost < cost:
            print("    - Acceptance probabilty = 1 as new_cost = {} < cost = {}...".format(new_cost, cost))
            return 1
        else:
            p = np.exp(- (new_cost - cost) / temperature)
            print("    - Acceptance probabilty = {:.3g}...".format(p))
            return p
            
    def temperature(fraction):
        """ Example of temperature dicreasing as the process goes on."""
        return max(0.01, min(1, 1 - fraction))
    #-----------------------------------------------------------------------------------------------
    def annealing(random_start,
                cost_function,
                random_neighbour,
                acceptance,
                temperature,
                maxsteps=100,
                debug=True,
                alpha=1,
                beta=1,
                gamma=1):
        """ Optimize the black-box function 'cost_function' with the simulated annealing algorithm."""
        state = random_start()
        cost = cost_function(state,alpha,beta,gamma)
        costs = [cost]
        states=[state[:]]
        for step in range(maxsteps):
            fraction = step / float(maxsteps)
        # fraction = step / float(100)
            print(fraction)
            T = temperature(fraction)
            global last_i
            global new_cost
            new_state,last_i = random_neighbour(state[:], T,cost,new_cost)
            new_cost = cost_function(new_state,alpha,beta,gamma)

            states.append(state[:])
            costs.append(cost)

            if debug: print(f"Step #{step}/{maxsteps} : T = {T:.3f}, state = {state}, cost = {cost:.3f}, new_state = {new_state}, new_cost = {new_cost:.3f} ...")
            if acceptance_probability(cost, new_cost, T) > rn.random():
                state, cost = new_state, new_cost
                print("  ==> Accept it!")
            else:
                print("  ==> Reject it...")
        return state, cost_function(state,alpha,beta,gamma), states, costs

        #------------------------------------------------------------------------
    #...
    state, c, states, costs=annealing(random_start, cost_function, random_neighbour, acceptance_probability, temperature, maxsteps=Max_steps, debug=True,alpha=alpha,beta=beta,gamma=gamma);
    states.append(state[:])
    states=states[2:]
    costs.append(c)
    costs=costs[2:]
    #------------------
    #-----COMPUTE ACCURACY AFTER SIMULATED ANNEALING
    fxp_by_layer=[]
    for i in range(len(list(state))):
        fxp_by_layer.append(Fxp(None, signed=True, n_int=n_int, n_frac=state[i]))
    for i, layer in enumerate(layers_of_interest):
        model.get_layer(layer.name).set_weights([Fxp(w_dict[layer.name][0], like=fxp_by_layer[i] ),Fxp(w_dict[layer.name][1], like=fxp_by_layer[i])])

    actual_acc= model.evaluate(train_images,  train_labels, verbose=0)[1]
    #----------------------- 

    def see_annealing(states, costs):
        plt.figure()
        plt.suptitle("Evolution of states and costs of the simulated annealing")
        plt.subplot(121)
        plt.plot(np.mean(states,1), 'r')
        plt.title(f"States      Final state: {state} -> Avg bits: {sum(state)/len(state):.3f}")
        plt.xlabel('Step')
        plt.ylabel('Avg nº of bits')
        plt.subplot(122)
        plt.plot(costs, 'b',label=f"\u03B1:{alpha:.2f}, \u03B2:{beta:.2f}, \u03B3:{gamma:.2f}\nLower bound: {lower_bound/factor:.3f}\nFinal acc: {actual_acc:.3f}\nError: {((lower_bound/factor)-actual_acc):.3f} ")
        plt.title(f"Costs           Final cost: {c:.3f}")
        plt.xlabel('Step')
        plt.ylabel('Cost')
        plt.legend()
        plt.savefig(f'./{simulations}_sims_max_steps_{Max_steps}/sim_{Max_steps}_steps_{time_stamp}.pdf', format="pdf", bbox_inches="tight")
        plt.close()
    # plt.show()
    def see_weights_cost(weights_cost):
        
        alpha_cost=[item[2] for item in weights_cost]
        beta_cost=[item[1] for item in weights_cost]
        gamma_cost=[item[0] for item in weights_cost]

        plt.figure()
        plt.plot(alpha_cost,label="\u03B2*lower_bound")
        plt.plot(beta_cost,label="\u03B1*avg_bits")
        plt.plot(gamma_cost,label="\u03B3*(lower_bound-actual_acc)")
        plt.legend()
        plt.savefig(f'./{simulations}_sims_max_steps_{Max_steps}/sim_weights_{Max_steps}_steps_{time_stamp}.pdf', format="pdf", bbox_inches="tight")
        plt.close()
    # plt.show()

    see_annealing(states, costs)
    see_weights_cost(weights_cost)
    final_acc_sim.append(actual_acc)
    final_avg_sim.append(sum(state)/len(state))
if(len(final_avg_sim)>1):
    #-------------- avg_bits
    mean_avg=sum(final_avg_sim)/len(final_avg_sim)
    std_avg=np.std(final_avg_sim)

    s_avg = np.random.normal(mean_avg, std_avg, 1000)
    count, bins, ignored = plt.hist(s_avg, 50, density=True,color='r',alpha=0.3)
    plt.plot(bins, 1/(std_avg * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - mean_avg)**2 / (2 * std_avg**2) ),
            linewidth=2, color='r',label=f"μ:{mean_avg:.3f}, \u03c3:{std_avg:.3f} \n\u03B1:{alpha:.2f}, \u03B2:{beta:.2f}, \u03B3:{gamma:.2f}\n")
    plt.title(f"Distribution of the final avg bits with {simulations} simulations ")
    plt.legend()
    plt.savefig(f'./{simulations}_sims_max_steps_{Max_steps}/distribution_avg_{Max_steps}_steps_{time_stamp}.pdf', format="pdf", bbox_inches="tight")

    plt.close()
    #--------------accuracy
    mean_acc=sum(final_acc_sim)/len(final_acc_sim)
    std_acc=np.std(final_acc_sim)

    s_acc = np.random.normal(mean_acc, std_acc, 1000)
    count, bins, ignored = plt.hist(s_acc, 50, density=True,color='b',alpha=0.3)
    plt.plot(bins, 1/(std_acc * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - mean_acc)**2 / (2 * std_acc**2) ),
            linewidth=2, color='b',label=f"μ:{mean_acc:.3f}, \u03c3:{std_acc:.3f} \n\u03B1:{alpha:.2f}, \u03B2:{beta:.2f}, \u03B3:{gamma:.2f}\n")
    plt.axvline(x=lower_bound/factor,label=f"Lower bound: {lower_bound/factor:.3f}",color='k',linestyle='dashed',linewidth=1)
    plt.title(f"Distribution of the final accuracy with {simulations} simulations ")
    plt.legend()
    plt.savefig(f'./{simulations}_sims_max_steps_{Max_steps}/distribution_acc_{Max_steps}_steps_{time_stamp}.pdf', format="pdf", bbox_inches="tight")
    plt.close()
    #-------------- accuracy
    with open(f'./{simulations}_sims_max_steps_{Max_steps}/final_acc_sim_{simulations}_sims_max_steps_{Max_steps}.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(final_acc_sim, file)
        file.close()

    with open(f'./{simulations}_sims_max_steps_{Max_steps}/final_avg_sim_{simulations}_sims_max_steps_{Max_steps}.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(final_avg_sim, file)
        file.close()