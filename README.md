# **Predicting Crop Disease through Image Classification**
# Capstone Project
Image classification to determine crop and healthy/disease based on images of crop leaves.

## Ben Geissel

### Introduction:
This project started with a question: How can food stability be improved? This question lead me to an image classification problem for diseased crop leaves.

Why are are plants/crops important towards food stability? Plants account for 80% of global human food consumption. Plant pests and diseases cause up to 40% of global food loss and this number is rising due to the increase in global trade which makes it easier for pests and diseases to leave their native environments. Additionally, given current population trends, demand for food is set to double by 2050. Increased populations will take up more space than ever before, even though we currently devote 50% of our land to agriculture.

I found the PlantVillage crop leaf image dataset online through a paper by David Hughes. This dataset includes 54K+ images of crop leaves that are from 38 separate classes.

### Goals:
Run numerous classification models to determine which of 38 classes an image of a crop leaf belongs to.
- Fit Numerous Machine Learning Classification Models with scikit learn (Naive Bayes, Random Forest, etc...)
- Fit a Convolutional Neural Network with Keras and utilize different types of layers
- Maximize accuracy across validation and testing sets of images
- Utilize Google Cloud computing to run jupyter notebooks with higher CPUs and GPUs
- Create prototype of possible application to use model in real world sense using Python Dash

### Tasks:
- ETL on image data
- Data exploration and visualizations
- Fit Naive Bayes and Random Forest Models
- Attempt SMOTE, PCA, SVM and other preprocessing/models, however data size and computing power wont allow for it
- Create confusion matrices and calculate other performance metrics
- Design and fit Covolutional Neural Network with Keras
- Evaluate model on validation and testing accuracy
- Save model
- Design prototype of possible web application to use model with Python Dash
- Test application

### Summary of Included Files:
The following files are included in the Github repository:
- ML_Crop_Disease_Model.ipynb
    - Jupyter Notebook consisting of data exploration and general machine learning models
          - Naive Bayes
          - Random Forest
    - PEP 8 Standards
    - Utilizes image_processing.py file for image preprocessing functions
    - Data importation, data cleaning, visualizations, machine learning models, confusion matrices, performance metrics
- DL_CNN_Crop_Disease_Model.ipynb
    - Jupyter Notebook consisting of deep learning model
          - Convolutional Neural Network
    - PEP 8 Standards
    - Utilizes image_processing.py file for image preprocessing functions
    - Data importation, data cleaning, standardization, neural network architecture, model fitting, performance metrics
- image_processing.py
   - Functions to help with image preprocessing
   - Image file to array
   - Flattening array
   - Normalizing pixel values
- app.py
   - Create prototype of possible application to use model in real world sense using Python Dash
   - Design dashboard appearance
   - Utilize callbacks to run functions based on user input
   - Locally hosted
- Crop_Disease_Leaf_Image_Classification_Model_Presentation.pdf
   - PDF presentation for project
   - Link to locally hosted application
- PlantVillage-Dataset folder
   - Folder containing all image data
- .gitignore

### Possible Next Steps:
In the future there are many next steps for this project. Three ideas to follow:
1. Deploy application to host on real website
2. Update application to take in picture of user and run this through the model
3. Gather more data and create crop specific models
