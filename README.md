DATA SET https://github.com/caige13/Diagnose-Breast-Cancer-Neural-Network
NOTE: TENSORFLOW AND KERAS ARE USED FOR NEURAL NETWORK.
The warnings and information status that Tensorflow gives I cannot figure out how to turn off because I havnt worked with tensor flow before. I chose to use Keras and Tensorflow because they are popular for machine learning and its rather easy to get it to use a GPU. For this project I do not use the GPU version of Tensorflow since I dont not know what type of environment this will be ran in.

The Program will error out if the file "Output_Table" is present because it can not overwrite it, delete it, etc. 
I put a check at the beginning of the program to remind you of this before going through the entire training process
only to have it error out on you. I also noticed if you have one of the graphs opened up when it tries to overwrite
it will error out because its being used. I also have a reminder to close all graphs so that a similar issue will not 
happen. The reminder just uses input() to pause the program, giving you time to close graphs. You can hit any key to continue it.

The my_model files are not necessary for the user. Those files are just used for Keras to reload initial weights to have a fresh start when reusing a neural network.

The Data Info folder contains information about the data being used and the correlation matrix. The data that I extracted from the website does not provide all the attributes names, so unfortunaetly I give them temporary column names. However, I do know that all 31 attributes are features of the cell in question which can be found in the wdbc NAMES file.

Under the Graph folder there is another folder named "Another breast Cancer Dataset" this was from an initial data set I was using that have 1-10 values in each column that ended up not working well with either normalization or standardization so I swapped to its sister data set found at the same link. The sister dataset did not have as friendly information about it but had more attributes, and due to it not being friendly I couldnt give a similar ananlysis. However, I kept this data in here because it still shows some correlation between cell information and the class. 

Aside from this information I use the following as Hyper Parameters:
	- Test Size: Represents the amount of data to use for the test set
	- Learning rate,
	- epochs,
	- number of hidden layers: Max 3 different inputs for simplicity,
	- unit count: The number of neurons for each layer ex: if num of hidden layer = 2 then [16, 8] for same index location of hidden layer and unit count.
	- activation function: the activation functions should be the loss activation functions that keras uses. https://keras.io/api/losses/
	- error function: This is used for optimization of the network. There is a enum and every error function in the enum will be ran inside the train_helper method.
The program will do every combination of the above hyper parameters, thus becareful how many you choose to use because it can end up taking a long time.

MODIFICATIONS:
	- History: I made a history object to keep track of all the attributes I want to track per trained model for easy analysis at the end.
	- Err enum: I made a enumerator for the different error functions for optimization to use. I can add more but it add to more computational time, and I believed comparing probability like binary crossentropy with MSE would be interesting.
	- NeuralNet: 
		- Add correlation matrix method to just print a correlation matrix at a given path.
		- preprocess: add parameters to allow customization of what to do with data such as what columns to exclude, what rows to delete if a certain value is found in it, and what columns to convert from categorical to numerical. Ontop of this it can do normalization or standardization.
		- train_evaluate: Add unit_count, and test_sizes as hyper parameters. There is commented out version also if you want to try other combinations. There is a format to follow with num_hidden_layers and unit_count which is that the indices must correlate.
		- train helper: private method that train_evaluate uses to actually create the neural network and train it with the given hyper parameters.
		- OVERALL: I tried to add abstraction to make the project more scalable for more parameters and potentially different data sets, but due to time constraint its not tested for other dataset and the project only wants 1.