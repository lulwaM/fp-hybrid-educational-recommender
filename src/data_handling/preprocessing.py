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

def merge_datasets(datasets):
    # COMMON DATASET IN ALL: courses because it is central to each student/assessment/vle resource

    # first merged is student info (demographic) / registration / course info
    # merge key are shared between all, left join used to maintain all student records, will deal with missing later
    student_data = datasets["student_info"].merge(datasets["student_registration"], on=["id_student","code_module","code_presentation"], how="left")

    student_data = student_data.merge(datasets["courses"], on=["code_module","code_presentation"], how="left")

    # second merged is student specific assessment data + general assessment info + standard course
    assessment_data = datasets["student_assessment"].merge(datasets["assessments"], on="id_assessment", how="left")
    assessment_data = assessment_data.merge(datasets["courses"], on=["code_module","code_presentation"], how="left")


    # third merged is student specific VLE data + general VLE info + standard course
    vle_data = datasets["student_vle"].merge(datasets["vle"], on=["id_site","code_module","code_presentation"],how="left")
    vle_data = vle_data.merge(datasets["courses"], on=["code_module","code_presentation"], how="left")

    #final dataset is simply the vle resources, needed for final recommendations
    resource_data = datasets["vle"].copy()

    # wayy more organized, reduced 7 DFs to 4
    merged_data = {"student_data": student_data, "assessment_data":assessment_data,"vle_data":vle_data, "resource_data": resource_data}

    return merged_data

def clean_datasets(datasets):

    # not directly editing variable, makign copy of dictionary via loop (good practice)
    cleaned_datasets = {key:dataframe.copy() for key,dataframe in datasets.items()}

    # 1. handling duplicate values
    cleaned_datasets["student_data"] = cleaned_datasets["student_data"].drop_duplicates()
    cleaned_datasets["assessment_data"] = cleaned_datasets["assessment_data"].drop_duplicates()

    # will not drop duplicates in vle as it indicates multiple interactions of student with resources, will be aggregated

    # 2. handling missing values

    #starting with student data dataframe:
    #  for imd band, will fill with mode because it is categorical data and it does not introduce bias
    cleaned_datasets["student_data"]["imd_band"] = cleaned_datasets["student_data"]["imd_band"].fillna(cleaned_datasets["student_data"]["imd_band"].mode()[0])

    # for date registration, will drop because its only 45 records
    cleaned_datasets["student_data"] = cleaned_datasets["student_data"].dropna(subset=['date_registration'])

    # from research paper on dataset, date unregistration:
    # Students, who completed the course have this field empty. 
    # Students who unregistered have Withdrawn as the value of the final_result in the studentInfo table.
    # solution: create separate column with withdrawn boolean flag and fill date unregistration with the module length
    cleaned_datasets["student_data"]["withdrew"] = cleaned_datasets["student_data"]["final_result"] == "Withdrawn"
    cleaned_datasets["student_data"]["date_unregistration"] = cleaned_datasets["student_data"]["date_unregistration"].fillna(cleaned_datasets["student_data"]["module_presentation_length"])


    # next the assessment data dataframe:

    # score will be dropped, very few records and risky to introduce bias in important feature
    cleaned_datasets["assessment_data"] = cleaned_datasets["assessment_data"].dropna(subset=['score'])

    #for missing dates, according to paper:
    #If the information about the final exam cut-off day is missing, it takes place during the last week of the module-presentation.
    cleaned_datasets["assessment_data"]["date"] = cleaned_datasets["assessment_data"]["date"].fillna(cleaned_datasets["assessment_data"]["module_presentation_length"]-7)

    #dataframe for vle:
    # large nubmer of week from/to missing, will remove columns entirely as they are not critical to recommendation
    cleaned_datasets["vle_data"] = cleaned_datasets["vle_data"].drop(columns=['week_from','week_to'])

    #last dataset for missing resources

    # large nubmer of week from/to missing, will remove columns entirely as they are not critical to recommendation
    cleaned_datasets["resource_data"] = cleaned_datasets["resource_data"].drop(columns=['week_from','week_to'])

    return cleaned_datasets


def typecasting_datasets(datasets):

    # not directly editing variable, makign copy of dictionary via loop (good practice)
    typecast_datasets = {key:dataframe.copy() for key,dataframe in datasets.items()}

    #first is student data

    # convert categories columns:
    student_categorical_columns = ["code_module","code_presentation","gender","region","highest_education","imd_band","age_band","final_result"]

    for column in student_categorical_columns:
        typecast_datasets["student_data"][column] = typecast_datasets["student_data"][column].astype("category")

    #convert booleans:
    typecast_datasets["student_data"]["withdrew"] = typecast_datasets["student_data"]["withdrew"].astype(bool)

    #code inspiration from :https://stackoverflow.com/questions/43897296/pandas-converting-yes-no-to-true-false-failing
    # replace y/n with true/false before typecast
    typecast_datasets["student_data"] =typecast_datasets["student_data"].replace({'disability': {'Y': True, 'N': False}})
    #end inspired code

    typecast_datasets["student_data"]["disability"] = typecast_datasets["student_data"]["disability"].astype(bool)

    #second is assessment data

    # convert categories columns:
    assessment_categorical_columns = ["code_module","code_presentation","assessment_type"]

    for column in assessment_categorical_columns:
        typecast_datasets["assessment_data"][column] = typecast_datasets["assessment_data"][column].astype("category")

    #convert boolean
    typecast_datasets["assessment_data"]["is_banked"] = typecast_datasets["assessment_data"]["is_banked"].astype(bool)

    # vle data to typecast

    # convert categories columns:
    vle_categorical_columns = ["code_module","code_presentation","activity_type"]

    for column in vle_categorical_columns:
        typecast_datasets["vle_data"][column] = typecast_datasets["vle_data"][column].astype("category")

    #final resources data types
    # convert categories columns:
    resource_categorical_columns = ["code_module","code_presentation","activity_type"]

    for column in resource_categorical_columns:
        typecast_datasets["resource_data"][column] = typecast_datasets["resource_data"][column].astype("category")

    return typecast_datasets

def processing_datasets(datasets):

    # not directly editing variable, makign copy of dictionary via loop (good practice)
    processed_datasets = {key:dataframe.copy() for key,dataframe in datasets.items()}

    #first for student data 

    #obtain registration duration
    processed_datasets["student_data"]["registration_duration"] = processed_datasets["student_data"]["date_unregistration"] - processed_datasets["student_data"]["date_registration"]

    #next for assessment data

    #average score of student for course across all assessments
    average_scores = processed_datasets["assessment_data"].groupby(["id_student", "code_module","code_presentation"])["score"].mean()

    #will rename index so that its clear when merged with final df
    average_scores = average_scores.reset_index(name="average_score")
    
    #merge with final
    processed_datasets["assessment_data"] = processed_datasets["assessment_data"].merge(average_scores, on=["id_student", "code_module","code_presentation"],how="left")

    #resource text used for final resource data for content based filtering
    processed_datasets["resource_data"]["resource_text"] = processed_datasets["resource_data"]["code_module"].astype(str) + " " + processed_datasets["resource_data"]["code_presentation"].astype(str) + " " + processed_datasets["resource_data"]["activity_type"].astype(str)

    #no feature creation / further processing required for vle data

    #code inspired by: https://www.kaggle.com/code/veenajoe/veena-vit-project-msc-ds-june-2026#4.2-Aggregating-students-based-on-id

    #final feature creation for vle data, will create new dataframe for aggregate interactions based on it

    #grouped so each row has one student and their intercation with 1 module resource, each aggregation column specifies 
    # new columnName =  (column from VLE data used in aggregation, aggregation function)
    #main feature used for collaboartive filtering is the total resource clicks
    interaction_data = processed_datasets["vle_data"].groupby(["id_student","code_module","code_presentation","id_site","activity_type"]).agg(total_resource_clicks=("sum_click","sum"),first_used=("date","min"),last_used=("date","max")).reset_index()
    #end inspired code

    #total dfs used will be 5
    processed_datasets["interaction_data"] = interaction_data

    return processed_datasets

