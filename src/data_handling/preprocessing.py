# for data storage
import pandas as pd

def load_datasets():
    
    # dictionary, key=file name value=its dataframe
    datasets = {}

    #7 files
    datasets["assessments"] = pd.read_csv("data/raw/assessments.csv")
    datasets["courses"] = pd.read_csv("data/raw/courses.csv")
    datasets["student_assessment"] = pd.read_csv("data/raw/studentAssessment.csv")
    datasets["student_info"] = pd.read_csv("data/raw/studentInfo.csv")
    datasets["student_registration"] = pd.read_csv("data/raw/studentRegistration.csv")
    datasets["student_vle"] = pd.read_csv("data/raw/studentVle.csv")
    datasets["vle"] = pd.read_csv("data/raw/vle.csv")

    return datasets


datasets = load_datasets()

for key,value in datasets.items():
    print(key)
    print(value.head(2))

