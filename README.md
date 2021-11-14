# Disaster Response Pipeline Project

## Table of Contents

1. [Motivation](#motivation)
2. [File description](#file)
3. [Results](#results)
4. [Instructions](#instructions)
5. [Acknowledgements](#acknowledgements)

## Motivation <a name="motivation"></a>

The aim of the project was structured into 3 sections:
1. Create an ETL-script that takes two csv-files, merges and clean them and stores the result into a SQLite database.
2. Create a Machine-Learning-script that takes the data from the database and builds a machine learning pipeline that 
processes text and then performs a multi-output-classification. The script uses a tokenize function to tokenize, normalize 
and lemmatize text by using NLTK.
3. Deploy the trained model to a Flask web-app that classifies user input for 36 different categories.

## File description <a name="file"></a>

The data that have been used for this supervised learning project were a dataset with disaster message-data 
(disaster_messages.csv) and the corresponding category-data (disaster_categories.csv). The datasets were provided by 
[Figure Eight](https://appen.com/).

- ETL Pipeline Preparation.ipynb: Jupyter notebook to build the ETL-script and getting an overview about the data.
  - data/process_data.py: Python-script built with the preparation above to create the pipeline.

- ML Pipeline Preparation.ipynb: Jupyter notebook to prepare the machine-learning-pipeline, create the tokenize function 
and optimize the model.
  - models/train_classifier.py: Python-script to create the machine-learning-pipeline.
  - models/classifier.pkl: Pickle-file containing the model trained in the pipeline above.

## Results <a name="results"></a>

As mentioned the result of this project is web-app where in case of an emergency a new message can be classified in 
different categories.


Potential for improvements: Although the ML-pipeline uses GridSearchCV to optimize the classification, there is still a 
lot of space for improvements:
- Test different estimators to optimize the classification (in this project RandomForestClassifier() has been used)
- Set an extended list of parameters for GridSearchCV for optimizing the model
- Use FeatureUnion to extend the pipeline for better results

## Instructions <a name="instructions"></a>

This instruction were originally made by the team at [Udacity's](https://www.udacity.com/).
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Acknowledgments <a name="acknowledgments"></a>

I would like to thank the team from [Udacity's](https://www.udacity.com/) for the great support and the brilliant online 
course [Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).