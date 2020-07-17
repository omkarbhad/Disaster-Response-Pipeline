# [Disaster Response Pipeline]() &middot; 



## Installation

* Check if [Python 3.8+ x64](https://www.python.org/downloads/) is installed.
* Create a Virtual Environment using ```python -m venv venv``` and activate it.
* Install the required libraries using ```pip install -r requirements.txt```


## Contents
```
- App
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- Data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- Models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```

### To Run the Code:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
