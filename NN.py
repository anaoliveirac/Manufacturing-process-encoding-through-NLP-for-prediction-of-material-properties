# -*- coding: utf-8 -*-
"""
Ana P. O. Costa et. al. "Manufacturing process encoding through natural language processing for prediction of material properties"
2023
"""
# necessary imports
import os
from urllib import request
from time import sleep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import trange, tqdm
from tqdm import tqdm
from scipy.stats import  stats
import itertools
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.model_selection import KFold
from torch import Tensor

def load_dataset(file_name):
    """
      loads the new dataset
    """

    if (os.path.isfile("steel_database.csv")):
        # import dataset.csv file int to a dataframe
        dataset_df = pd.read_csv(file_name, sep=';', low_memory=False)

         # create a data frame with the features to train:
        x_train = pd.DataFrame(dataset_df, columns=['Cr',	'Al',	'B',	'Co',	'Mo',	'Ni',	'Ti',	'Zr',
                                          'C',	'Fe',	'Pd',	'Mn',	'P',	'Si', 'N',	'Cu',	'Nb',	'Se',	'Ta',	'W',	'V',	'S',
                                          'e1',	'e2',	'e3',	'e4','e5', 'e6', 'e7',	'e8',	'e9',	'e10', 'e11', 'e12','e13','e14', 'e15',
                                          'e16',	'e17',	'e18',	'e19', 'e20', 'e21','e22','e23',
                                          'e24',	'e25',	'e26',	'e27',	'e28', 'e29','e30','e31',
                                          'e32', 'e33',	'e34',	'e35',	'e36',	'e37', 'e38', 'e39',
                                          'e40', 'e41', 'e42',	'e43',	'e44',	'e45',	'e46', 'e47',
                                          'e48','e49','e50', 'e51',	'e52',	'e53',	'e54',	'e55', 'e56',
                                          'e57','e58','e59', 'e60',	'e61',	'e62',	'e63',	'e64', 'e65',
                                          'e66','e67','e68', 'e69',	'e70',	'e71',	'e72',	'e73', 'e74',
                                          'e75','e76','e77', 'e78',	'e79',	'e80',	'e81',	'e82', 'e83',
                                           'e84','e85','e86', 'e87',	'e88',	'e89',	'e90',	'e91', 'e92',
                                            'e93','e94','e95', 'e96',	'e97',	'e98',	'e99',	'e100', 'e101',
                                           'e102','e103','e10', 'e105', 'e106','e107','e108', 'e109', 'e110',
                                          'e111','e112','e113', 'e114','e115',	'e116',	'e117',	'e118',	'e119',
                                          'e120', 'e121','e122', 'e123', 'e124', 'e125', 'e126', 'e127',
                                         'e128', 'e129', 'e130', 'e131','e132', 'e133', 'e134', 'e135','e136', 'e136', 'e137'
                                         'e135','e136',	'e137',	'e138',	'e139',	'e140',	'e141',	'e142',	'e143',	'e144',
                                         'e145','e146',	'e147',	'e148',	'e149',	'e150',	'e151',	'e152',	'e153',	'e154',
                                         'e155','e156',	'e157',	'e158',	'e159',	'e160',	'e161',	'e162',	'e163',	'e164',
                                         'e165','e166',	'e167',	'e168',	'e169',	'e170',	'e171',	'e172',	'e173',	'e174',
                                         'e175','e176',	'e177',	'e178',	'e179',	'e180',	'e181',	'e182',	'e183',	'e184',
                                         'e185','e186',	'e187',	'e188',	'e189',	'e190',	'e191',	'e192',	'e193',	'e194',
                                         'e195','e196',	'e197',	'e198',	'e199',	'e200',	'e201',	'e202',	'e203',	'e204'])
        #To change the one hot encode to label encode, should be deleted  'e1" to 'e204' and add 'label' to columns.
        # Choose the target variable
        y_train = pd.DataFrame(dataset_df, columns=['UTS'])        
        #y_train = pd.DataFrame(dataset_df, columns=['YS'])
        #y_train = pd.DataFrame(dataset_df, columns=['Elongation'])

        # convert to float32 type
        y_train = y_train.astype(np.float32)
        x_train = x_train.astype(np.float32)

        # remove nan values covert to zero
        x_train = x_train.replace(np.nan, 0)
        y_train = y_train.replace(np.nan, 0)

        # save the dataframe to csv file
        x_train.to_csv('x_train_v2.csv', sep=';', index=False)
        y_train.to_csv('y_train_v2.csv', sep=';', index=False)

        # convert dataframe to numpy array
        x_train = x_train.values
        y_train = y_train.values

    else:
        raise

    return x_train, y_train

class FullyConnectedNN(torch.nn.Module):
    """ fullyconnected multilayer feedforward topology """

    def __init__(self, input_size, device='cpu'):

        # This function creates an instance of the base nn.Module class
        super(FullyConnectedNN, self).__init__()

        self.device = device
        
        # Define NN sctruture and proportion or neurons to dropout
        
        #Structure 1
        # Create the input layer of the neural network, with
        self.input_layer = torch.nn.Linear(input_size, 128)
        self.hidden_layer1 = torch.nn.Linear(128, 64)
        self.drop1=nn.Dropout(p=0.25)
        self.hidden_layer2 = torch.nn.Linear(64, 32)
        self.drop2=nn.Dropout(p=0.25)
        #self.drop2=nn.Dropout(p=0.5)
        self.hidden_layer3 = torch.nn.Linear(32, 16)
        self.drop3=nn.Dropout(p=0.25)
        #self.drop2=nn.Dropout(p=0.5)

        self.output_layer = torch.nn.Linear(16, 1)
        
        #Structure 2
        # Create the input layer of the neural network, with
        #self.input_layer = torch.nn.Linear(input_size, 64)
        #self.hidden_layer1 = torch.nn.Linear(64, 32)
        #self.drop1=nn.Dropout(p=0.25)
        #self.hidden_layer2 = torch.nn.Linear(32, 16)
        #self.drop2=nn.Dropout(p=0.25)
        #self.hidden_layer3 = torch.nn.Linear(16, 8)
        #self.drop3=nn.Dropout(p=0.25)

        #self.output_layer = torch.nn.Linear(8, 1)
        
        #Structure 3
        # Create the input layer of the neural network, with
        #self.input_layer = torch.nn.Linear(input_size, 124)
        #self.hidden_layer1 = torch.nn.Linear(124, 77)
        #self.drop1=nn.Dropout(p=0.25)
        #self.hidden_layer2 = torch.nn.Linear(77, 38)
        #self.drop2=nn.Dropout(p=0.25)
        #self.hidden_layer3 = torch.nn.Linear(38, 16)
        #self.drop3=nn.Dropout(p=0.25)
        #self.output_layer = torch.nn.Linear(16, 1)
        #self.output_layer = torch.nn.Linear(8, 1)

      


    def forward(self, x):
        # define the forward pass
        x = torch.nn.functional.relu(self.input_layer(x))
        x = torch.nn.functional.relu(self.hidden_layer1(x))
        x = torch.nn.functional.relu(self.hidden_layer2(x))
        x = torch.nn.functional.relu(self.hidden_layer3(x))
        x = self.output_layer(x)

        return x

def model_training(model, x_train, y_train,
                   epochs=100,
                   learning_rate=0.001,
                   device='cpu'):
    """ model training """

    # define the loss function
    criterion = torch.nn.SmoothL1Loss()


    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # convert the data to a tensor
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

    # move the model to the device
    model.to(device)

    model.train()

    loss = 0
    loss_list = []
    print('model training...')
    with tqdm(total=epochs, colour='red') as pbar:

        # loop over the number of epochs
        for epoch in range(epochs):

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            y_predicted_tensor = model(x_train_tensor)

            # calculate the loss
            loss = criterion(y_predicted_tensor, y_train_tensor)

            # backward pass
            loss.backward()

            # update the parameters
            optimizer.step()

            # append the loss to the loss array
            loss_list.append(loss.item())

            # convert to cpu tensor
            y_predicted_tensor = y_predicted_tensor.cpu()

            # convert to numpy array
            preditions = y_predicted_tensor.detach().numpy()

            # calculate the accuracy of the model
            accuracy = np.mean(np.abs(preditions - y_train))

            pbar.update(1)
            pbar.set_description(f"Epoch {epoch}")

            pbar.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
            sleep(0.01)

    # print the loss
    print("Training Loss: {:.4f}%".format(loss.item()))
    # convert the model to a numpy array
    y_predicted = y_predicted_tensor.cpu()
    y_predicted = y_predicted.detach().numpy()

    return model, loss_list, y_predicted, loss

def model_inference(model, x_test, y_test, device='cpu'):
    """ Test the model """

    # create the loss function
    criterion = torch.nn.SmoothL1Loss()

    y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)

    # move the model to the device
    model.to(device)

    # predict the y_test
    y_predicted_tensor = model(x_test_tensor)

    # calculate the loss
    loss = criterion(y_predicted_tensor, y_test_tensor)

    # print the loss
    print("Inference Loss: {:.4f}%".format(loss.item()))

    # convert the model to a numpy array
    y_predicted_tensor = y_predicted_tensor.cpu()

    # convert the model to a numpy array
    y_predicted = y_predicted_tensor.detach().numpy()

    return model, y_predicted, loss

def plot_predicted_vs_actual(y_predicted, y_actual, title='Predicted Vs Actual'):

    # plot the y_predicted vs y_test
    plt.title(title)
    plt.plot(y_predicted, color='darkgreen', label='predicted UTS', linewidth=2.0)
    plt.plot(y_actual, color='orange', label='actual UTS', linewidth=2.0)
    #plt.plot(y_predicted, color='darkgreen', label='predicted YS', linewidth=2.0)
    #plt.plot(y_actual, color='orange', label='actual YS', linewidth=2.0)
    #plt.plot(y_predicted, color='darkgreen', label='predicted Elongation', linewidth=2.0)
    #plt.plot(y_actual, color='orange', label='actual Elongation', linewidth=2.0)
    # axis labels
    plt.xlabel('Samples')
    plt.ylabel('UTS (MPa)')

    # create a legend in upper left corner
    plt.legend(loc='upper left')
    # show the plot
    plt.show()
    plt.close()

def plot_scatter_plot(y_predicted, y_actual, title='Predicted Vs Actual'):

    # plot the y_predicted vs y_test
    plt.title(title)

    # create the axis with the same range as the data
    plt.xlim(min(y_predicted), max(y_predicted))
    plt.ylim(min(y_actual), max(y_actual))

    # create  a x_array for the x-axis within the range of data
    x_array = np.linspace(min(y_predicted), max(y_predicted), len(y_predicted))

    # create  a y_array for the y-axis within the range of data
    y_array = np.linspace(min(y_actual), max(y_actual), len(y_actual))

    # plot y_predicted in orange
    plt.scatter(y_actual, y_predicted,  color='darkgreen', label='predicted UTS', linewidth=2.0)

    plt.scatter(y_predicted, y_actual, color='orange', label='actual UTS', linewidth=2.0)
    
    #plt.scatter(y_actual, y_predicted,  color='darkgreen', label='predicted YS', linewidth=2.0)

    #plt.scatter(y_predicted, y_actual, color='orange', label='actual YS', linewidth=2.0)
    
    #plt.scatter(y_actual, y_predicted,  color='darkgreen', label='predicted Elongation', linewidth=2.0)

    #plt.scatter(y_predicted, y_actual, color='orange', label='actual Elongation', linewidth=2.0)

    # plot a linear line that matches y_predicted to y_actual
    plt.plot(x_array, y_array, color='red', label='actual=predicted', linestyle='--')



    # axis labels
    plt.xlabel('Predicted UTS (MPa)')
    plt.ylabel('Actual UTS (MPa)')
    
    #plt.xlabel('Predicted YS (MPa)')
    #plt.ylabel('Actual YS (MPa)')
    
    #plt.xlabel('Predicted Elongation (%)')
    #plt.ylabel('Actual Elongation (%)')

    # create a legend in upper left corner
    plt.legend(loc='upper left')
    # show the plot
    plt.show()
    plt.close()

def plot_model_loss(loss_list, epochs, title='Model - Loss Function'):
    """
      plot the loss function
    """

    plt.title(title)
    plt.plot(loss_list)
    plt.xlabel('epochs ' + str(epochs))
    plt.ylabel('loss')
    plt.show()
    plt.close()
    
# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    Pred = model(row)
    # retrieve numpy array
    Pred = Pred.detach().numpy()
    return Pred

# Define the device in use for training and inference
# If a Nvidia GPU with CUDA support is available
# the training will be done in a GPU, otherwise use the CPU will be used
if torch.cuda.is_available():
    print("GPU with CUDA Support is available")
    print("Using CUDA Device: {}".format(torch.cuda.device_count()))
    processing_unit = 'cuda'
else:
    print("CUDA GPU Not Available, using CPU")
    processing_unit = 'cpu'

# preprocess, if necessary and load the data to the dataset
x_train, y_train = load_dataset("steel_database.csv")

num_folds = 4
#num_folds= 2
#num_folds= 6
kfold = KFold(n_splits=num_folds, shuffle=True)
acc_per_fold = []
loss_per_fold = []
for train, test in kfold.split(x_train, y_train):

#x, y = x_train,y_train
#x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.15)

# create a instance of the model
    model = FullyConnectedNN(input_size=x_train.shape[1], device=processing_unit)

# if the model is not trained, train it
    if not os.path.isfile('models/model_fcnnv1.pt'):

    # train the model
      model, loss_list, y_predicted, loss = model_training(model,
                                                        x_train[train], y_train[train],
                                                        epochs=10000,
                                                        learning_rate=0.04,
                                                        device=processing_unit)

    # plot the predicted values Vs actual Values from model training
    plot_predicted_vs_actual(y_predicted, y_train,
                            title="Model Training - Predicted Vs Actual")



    plot_model_loss(loss_list, 10000, title="Model Training loss - SmoothL1Loss")


     # Infer the test data using the trained model
    model, y_predicted, loss_list = model_inference(
             model, x_train[test], y_train[test], device=processing_unit)

      # Plot the predicted values vs actual values from the model training
    plot_predicted_vs_actual(
          y_predicted, y_train[test], title="Model Inference - Predicted Vs Actual")

    plot_scatter_plot(y_predicted, y_train[test] , title="Model Inference - Scatter Plot", )

      # calculate the MSE
    mse = np.sqrt(mean_squared_error(y_train[test], y_predicted))

      # calculate the MAE
    mae = mean_absolute_error(y_train[test], y_predicted)

      # calculate the R2 squared
    r2_squared = r2_score(y_train[test], y_predicted)

    print ("FCNN MSE: {:.4f}%".format(mse))
    print ("FCNN MAE: {:.4f}%".format(mae))
    print ("FCNN R2 Squared: {:.4f}%".format(r2_squared))
    
    #Prediction of SDSS 25Cr-7Ni-Mo-N

row = [25.3,	0,	0,	0,	3.8,	7.9,	0,	0,	0.2,	60.5,	0,	0.7,	0,	0.7,	0.2,    0,	0,	0,	0,	0.7,    0,	0,
          0,	0,	0,	0,	  0,      1,	0,	0,	  0,	   0,	0,	  0,	0,	  0,	  0,	0,	0,	0,	0,	  0,	0,  0,
          0,	0,	0,	0,	  0,	  0,    0,	0,	  0,	   0,	0,	  0,	0,	  0,	  0,	0,	0,	0,	0,	  0,	0,	0,
       	  0,    0,	0,	0,	  0,  	  0,	0,	0,	  0,	   0,	0,	  0,	0,	  0,	  0,	0,	0,	0,	0,	  0,	0,	0,
          0,	0,	0,	0,	  0,	  0,	0,	0,	  0,	   0,	0,	  0,	0,	  0,	  0,	0,	0,	0,	0,	  0,	0,	0,
          0,	0,	0,	0,	  0,	  0,	0,	0,	  0,	   0,	0,	  0,	0,	  0,	  0,	0,	0,	0,	0,	  0,	0,	0,
          0,	0,	0,	0,	  0,	  0,	0,	0,	  0,	   0,	0,	  0,	0,	  0,	  0,	0,	0,	0,	0,	  0,	0,	0,
          0,	0,	0,	0,	  0,	  0,	0,	0,	  0,	   0,	0,	  0,	0,	  0,	  0,	0,	0,	0,	0,	  0,	0,	0,
          0,	0,	0,	0,	  0,	  0,	0,	0,	  0,	   0,	0,	  0,	0,	  0,	  0,	0,	0,	0,	0,	  0,	0,	0,
          0,    0,	0,  0,    0,	  0,    0,	0,	  0,	   0,   0,	  0,	0,	  0,	  0,	0,	0,	0,	0,	  0,    0,	0,	
          0,	0,	0,	0,    0,	  0,	0,	0,   0]
Pred = predict(row, model)

print('Predicted SDSS 25Cr-7Ni-Mo-N: %.3f' % Pred)

# Stainless Steel 32Cr-7Ni-Mo-N
row2 = [32,	0,	0,	0,	7,	7.9,	0,	0,	0.2,	50.78,	0,	0.7,	0,	0.7,	0.2,  0,	0,	0,	0,	0.7,  0,	0,
          0,	0,	0,	0,	  0,      1,	0,	0,	  0,	   0,	0,	  0,	0,	  0,	  0,	0,	0,	0,	0,	  0,	0,  0,
          0,	0,	0,	0,	  0,	  0,    0,	0,	  0,	   0,	0,	  0,	0,	  0,	  0,	0,	0,	0,	0,	  0,	0,	0,
       	  0,    0,	0,	0,	  0,  	  0,	0,	0,	  0,	   0,	0,	  0,	0,	  0,	  0,	0,	0,	0,	0,	  0,	0,	0,
          0,	0,	0,	0,	  0,	  0,	0,	0,	  0,	   0,	0,	  0,	0,	  0,	  0,	0,	0,	0,	0,	  0,	0,	0,
          0,	0,	0,	0,	  0,	  0,	0,	0,	  0,	   0,	0,	  0,	0,	  0,	  0,	0,	0,	0,	0,	  0,	0,	0,
          0,	0,	0,	0,	  0,	  0,	0,	0,	  0,	   0,	0,	  0,	0,	  0,	  0,	0,	0,	0,	0,	  0,	0,	0,
          0,	0,	0,	0,	  0,	  0,	0,	0,	  0,	   0,	0,	  0,	0,	  0,	  0,	0,	0,	0,	0,	  0,	0,	0,
          0,	0,	0,	0,	  0,	  0,	0,	0,	  0,	   0,	0,	  0,	0,	  0,	  0,	0,	0,	0,	0,	  0,	0,	0,
          0,    0,	0,  0,    0,	  0,    0,	0,	  0,	   0,   0,	  0,	0,	  0,	  0,	0,	0,	0,	0,	  0,    0,	0,	
          0,	0,	0,	0,    0,	  0,	0,	0,   0]
Pred2 = predict(row2, model)
print('Predicted Stainless Steel 32Cr-7Ni-Mo-N: %.3f' % Pred2)
