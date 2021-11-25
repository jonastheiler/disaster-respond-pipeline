# Disaster Response Pipeline Project

## Table of Contents

1. [Motivation](#motivation)
2. [File description](#file)
3. [Results](#results)
4. [Instructions](#instructions)
5. [Acknowledgements](#acknowledgements)

## Motivation <a name="motivation"></a>

The aim of the project was divided into 3 sections:
1. creating an ETL-pipeline that takes two CSV files, merges and cleans them, and stores the result in a SQLite database.
2. creating a ML-pipeline that takes the data from the database and processes text and performs a multi-output 
classification. The script uses NLTK, scikit-learn's Pipeline and GridSearchCV.
3. deploying the trained model in a Flask web app where you can input a new message and get classification results in 
different categories.


## File description <a name="file"></a>

The data used for this supervised learning project were a dataset of disaster responding data (disaster_messages.csv) 
and the corresponding category data (disaster_categories.csv). The datasets were provided by [Figure Eight](https://appen.com/).

The following files has been uploaded to the repository:

- app
  - run.py: python script to run the model for the web app.
  - templates
    - master.html: script to construct the web app.
    - go.html: extension for master.html.

- data
  - process_data.py: Python script created with the above preparation to create the pipeline.
  - disaster_message.csv: data input for the message data.
  - disaster_categories.csv: data input for the category data.
  - DisasterResponse.db: database containing the merged, cleaned data produced by the process_data.py script

- models
  - train_classifier.py: Python script to create the machine learning pipeline.
  - (missing, too big) classifier.pkl: Pickle file that contains the model from the train_classifier.py script.

- preparation
  - ETL Pipeline Preparation.ipynb: Jupyter notebook to create the ETL script and get an overview of the data.
  - ML Pipeline Preparation.ipynb: Jupyter notebook to prepare the machine learning pipeline, create the tokenize function.

- README.md


## Results <a name="results"></a>

As already mentioned, the result of this project is a web app that can be used to classify events into different 
categories so it will be possible to forward the message to the appropiate disaster relief agency.

![bild](Screenshot-Disaster-Response.png)
Screenshot of the Disaster Response Pipeline



#### Reflection

I am happy with the result and fully functional web app. Although everything works, there is still enough room for 
improvements:
- Testing different estimators to optimise the classification (RandomForestClassifier() was used in this project).
- Setting an extended list of parameters for GridSearchCV to optimise the model (long performance time problems)
- Using FeatureUnion to extend the pipeline with other transformations for better results. For example on could have 
used different transformations in NLP.
- Also, the web app could have been deployed to Heroku (or similar provider) for easy access to the result.

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