# machine learning coder and way to do it
#step 1 import data
# step 2 clean data
# 3 split data. Training set/test set
#4 create a model maybe import an algorithm from a library. usually this is how is done.
#5 check the output
#6 improve

# tools to use to do the previous steps
# numpy lists, arrays and multidimention arrays
# pandas data analysis
#scikit-learn for model part of proces (regression, classification etc.)
#matplotlib
#anaconda (jupyter notebook)
# kaggle has free data sets to play around




# creating an image recognition app

from imageai.Classification import ImageClassification
import os
import torch
import torchvision.models as models

# Load the pre-trained ResNet50 model
resnet50 = models.resnet50(pretrained=True)

# Save the model to a file
torch.save(resnet50.state_dict(), 'ResNet50.pth')


execution_path = os.getcwd()

# Initialize the ImageClassification object
prediction = ImageClassification()

# Set the model type to ResNet50 for PyTorch
prediction.setModelTypeAsResNet50()

# Set the model path to a PyTorch model file (.pth)
# Make sure you download the PyTorch model and place it in the specified path
prediction.setModelPath(os.path.join(execution_path, 'ResNet50.pth'))

#laod the model
prediction.loadModel()

# Make predictions on the specified image
predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, 'giraffe.jpg'), result_count = 5) #this count is the number of prediction we want the model to return

# Print the predictions and their corresponding probabilities
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, ' : ', eachProbability)

