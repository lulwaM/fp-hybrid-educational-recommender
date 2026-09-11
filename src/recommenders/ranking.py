#imports
import pandas as pd
import numpy as np

# ML ranking model will use ALL recommenders
from src.recommenders import popularity, collaborative, content_based

# for ML model creation/preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# for saving/loading model
import joblib

# generate possible recommendations for a student-course record, k = number of candidate recommendations to generate, must take course details to filter
def generate_candidates(id_student,interaction_data,resource_data, code_module,code_presentation, k=20, popularity_recs=None, collaborative_recs=None, content_recs=None):

    if(popularity_recs is None or collaborative_recs is None or content_recs is None ):
        # generate scores since not generated

        popularity_recs = popularity.popularity_scores(id_student,interaction_data)
        collaborative_recs = collaborative.collaborative_scores(id_student,interaction_data)
        content_recs = content_based.content_scores(id_student,resource_data,interaction_data)

    # filter to current course first then take top k
    popularity_recs = popularity_recs[(popularity_recs["code_module"] ==code_module) & (popularity_recs["code_presentation"]==code_presentation)]
    collaborative_recs = collaborative_recs[(collaborative_recs["code_module"] ==code_module) & (collaborative_recs["code_presentation"]==code_presentation)]
    content_recs = content_recs[(content_recs["code_module"] ==code_module) & (content_recs["code_presentation"]==code_presentation)]

    # limit recommendations to top k 
    popularity_recs = popularity_recs.head(k)
    collaborative_recs = collaborative_recs.head(k)
    content_recs = content_recs.head(k)

    # create candidates with resource details + each score (using details from populariyt scores then extracting scores from each)
    candidates = popularity_recs[["id_site","code_module","code_presentation","activity_type","popularity_score"]].copy()
    candidates = candidates.merge(collaborative_recs[["id_site","collaborative_score"]], on="id_site",how="outer")
    candidates = candidates.merge(content_recs[["id_site","content_score"]], on="id_site",how="outer")

    # since resource details comes from popularity only, may be missing from recommendations produced only by other scores (fix: use resource data)
    candidates = candidates.drop(columns=["code_module","code_presentation","activity_type"])
    candidates = candidates.merge(resource_data[["id_site","code_module","code_presentation","activity_type","resource_text"]],on="id_site",how="left")

    # outer joins may lead to NaN for some scores since not all recommenders recommend same resources, so fill with 0
    candidates[["popularity_score","collaborative_score","content_score"]] = candidates[["popularity_score","collaborative_score","content_score"]].fillna(0)

    # attach student id to their candidate recommendations
    candidates["id_student"] = id_student

    return candidates

# creates dataframe where for eah student = candidate recs generated from history interactions and targets generated from future interactions + student features attached
def create_ml_dataset(history_interactions, future_interactions,resource_data,student_features,k=20):

    #get dataframe of non-duplicate students in historica and future interactions separately
    historical_students = history_interactions[["id_student","code_module","code_presentation"]].drop_duplicates()
    future_students = future_interactions[["id_student","code_module","code_presentation"]].drop_duplicates()

    # only evaluate students with future interactions to evaluate via inner join
    evaluated_students = historical_students.merge(future_students, on=["id_student","code_module","code_presentation"],how="inner")

    candidate_rows=[]

    #iterate over each evaluated student (values are index,row where row is each evaluated student record with keys id/coures module/etc)
    #iteration code inspired by: https://stackoverflow.com/questions/16476924/how-can-i-iterate-over-rows-in-a-pandas-dataframe
    for index,evaluated_student in evaluated_students.iterrows():

        #only print after every 100 loops, prevents heavy console logging
        if(index % 100 == 0):
            print("creating candidate rows:",index)

    #end inspired code

        # generate candidate recommendations from historical interactions
        candidates = generate_candidates(evaluated_student["id_student"],history_interactions,resource_data,evaluated_student["code_module"],evaluated_student["code_presentation"],k)

        # make sure candidates only for this specific student-course record
        candidates = candidates[(candidates["id_student"]==evaluated_student["id_student"]) & 
                                                 (candidates["code_module"]==evaluated_student["code_module"]) & 
                                                 (candidates["code_presentation"]==evaluated_student["code_presentation"])]


        # no recommendations generated for this student-course, skip
        if (candidates.empty):
            continue

        #items from interactions, returns series
        history_items = history_interactions.loc[(history_interactions["id_student"]==evaluated_student["id_student"]) & 
                                                 (history_interactions["code_module"]==evaluated_student["code_module"]) & 
                                                 (history_interactions["code_presentation"]==evaluated_student["code_presentation"]),"id_site"]

        # resources student actually uses in future 
        future_items = future_interactions.loc[(future_interactions["id_student"]==evaluated_student["id_student"]) & 
                                                 (future_interactions["code_module"]==evaluated_student["code_module"]) & 
                                                 (future_interactions["code_presentation"]==evaluated_student["code_presentation"]),"id_site"]

        #removed used resources in history from future items, as goal is to recommend new items, filter series
        relevant_items = future_items[~future_items.isin(history_items)].unique()

        # create binary target for ML model based on if the canddiate recommendations are in future used resources, integers where 1=used 0=not used
        candidates["target"] =  candidates["id_site"].isin(relevant_items).astype(int)

        candidate_rows.append(candidates)

    # stacks the list of dataframes into single dataframe, where final return is a list of student-course recommendations + scores + binary target
    ml_dataset = pd.concat(candidate_rows, ignore_index=True)

    # attach student features to ml dataset, merge on the student/course identifiers while maintaining all recommendations via left join
    ml_dataset = ml_dataset.merge(student_features, on=["id_student","code_module","code_presentation"], how="left")


    return ml_dataset

def sample_training_data(training_dataset, sample_size=5000):
    # use stratified to maintain balanced classes
    # code inspired by: https://stackoverflow.com/questions/35472712/how-to-split-data-on-balanced-training-set-and-test-set-on-sklearn
    sample, _ = train_test_split(training_dataset,train_size=sample_size,stratify=training_dataset["target"],random_state=10)
    # end inspired code

    # remove indices because it is sampled
    return sample.reset_index(drop=True)


# prepare test/train X/y to feed to model
def prepare_model_input(training_dataset, test_dataset):

    # easy: y=target
    y_train = training_dataset["target"]
    y_test = test_dataset["target"]

    # now for x, select features to include for the model to learn from 
    # full list of features:
    # id_site,popularity_score,collaborative_score,content_score,code_module,code_presentation,activity_type,id_student,target,gender,region,highest_education,imd_band,age_band,num_of_prev_attempts,studied_credits,disability,date_registration,module_presentation_length,average_score,min_score,max_score,assessments_submitted,average_submission_date,average_assessment_weight,assessment_type_diversity,total_resource_clicks,average_resource_clicks,first_used,last_used,resources_used,interaction_duration,days_registered, resource_text

    # NOTE: identifiers removed like id site/id student because they are uninformative, no numeric meaning + data privacy
    # also, no ML specific preprocessing done yet. just choosing features
    feature_columns= [
        # scores, required
        "popularity_score","collaborative_score","content_score",
        # resource info, already includes code module/presentation/activity type so exclude that
        "resource_text", 
        # student profile info, excluded date registration and module presentation length because already have stronger features to learn from like days_registered and resource_text
        "gender","region","highest_education","imd_band","age_band", "num_of_prev_attempts","studied_credits","disability",
        # assessment features
        "average_score","min_score","max_score","assessments_submitted","average_submission_date","average_assessment_weight","assessment_type_diversity",
        # interaction features for VLE resources, all included because informative
        "total_resource_clicks","average_resource_clicks","first_used","last_used","resources_used","interaction_duration","days_registered"
        ]

    # extract columns for X data
    X_train = training_dataset[feature_columns].copy()
    X_test = test_dataset[feature_columns].copy()

    # handle missing values to prevent crash
    
    # for categorical columns, pd.NaN not accepted for missing, 
    categorical_features =[
        "gender","region","highest_education","imd_band","age_band", "disability",
    ]

    # so convert to np.nan after converting it to object for easy processing
    for feature in categorical_features:
        # code to convert np.nan inspired by: https://stackoverflow.com/questions/14162723/replacing-pandas-or-numpy-nan-with-a-none-to-use-with-mysqldb
        X_train[feature] = X_train[feature].astype(object).where(X_train[feature].notna(),np.nan)
        X_test[feature] = X_test[feature].astype(object).where(X_test[feature].notna(),np.nan)
        # end inspired code


    # text cannot be empty because then cannot be vectorized
    X_train["resource_text"] = X_train["resource_text"].fillna("").astype(str)
    X_test["resource_text"] = X_test["resource_text"].fillna("").astype(str)

    # return train/test X/y
    return (X_train, y_train, X_test, y_test)

# USED only in final API recommendations, copied from prepare model input but removed y part to keep X for feeding into model for predictions
def prepare_prediction_input(ML_dataset):

# copied from prepare model input but removed anything related to y (no need target)
    # for x, select features to include for the model to learn from 
    # full list of features:
    # id_site,popularity_score,collaborative_score,content_score,code_module,code_presentation,activity_type,id_student,target,gender,region,highest_education,imd_band,age_band,num_of_prev_attempts,studied_credits,disability,date_registration,module_presentation_length,average_score,min_score,max_score,assessments_submitted,average_submission_date,average_assessment_weight,assessment_type_diversity,total_resource_clicks,average_resource_clicks,first_used,last_used,resources_used,interaction_duration,days_registered, resource_text

    # NOTE: identifiers removed like id site/id student because they are uninformative, no numeric meaning + data privacy
    # also, no ML specific preprocessing done yet. just choosing features
    feature_columns= [
        # scores, required
        "popularity_score","collaborative_score","content_score",
        # resource info, already includes code module/presentation/activity type so exclude that
        "resource_text", 
        # student profile info, excluded date registration and module presentation length because already have stronger features to learn from like days_registered and resource_text
        "gender","region","highest_education","imd_band","age_band", "num_of_prev_attempts","studied_credits","disability",
        # assessment features
        "average_score","min_score","max_score","assessments_submitted","average_submission_date","average_assessment_weight","assessment_type_diversity",
        # interaction features for VLE resources, all included because informative
        "total_resource_clicks","average_resource_clicks","first_used","last_used","resources_used","interaction_duration","days_registered"
        ]

    # extract columns for X data
    X = ML_dataset[feature_columns].copy()

    # handle missing values to prevent crash
    
    # for categorical columns, pd.NaN not accepted for missing, 
    categorical_features =[
        "gender","region","highest_education","imd_band","age_band", "disability",
    ]

    # so convert to np.nan after converting it to object for easy processing
    for feature in categorical_features:
        # code to convert np.nan inspired by: https://stackoverflow.com/questions/14162723/replacing-pandas-or-numpy-nan-with-a-none-to-use-with-mysqldb
        X[feature] = X[feature].astype(object).where(X[feature].notna(),np.nan)
        # end inspired code


    # text cannot be empty because then cannot be vectorized
    X["resource_text"] = X["resource_text"].fillna("").astype(str)

    # return X to feed to model for predictions
    return X

# returns fitted ML model
def create_ML_model(X_training_data, y_training_data, model_type='random_forest'):


    # divide features into the 3 data types, different preprocessing for each

    numerical_features = [
        "popularity_score","collaborative_score","content_score",
        "num_of_prev_attempts","studied_credits",
        "average_score","min_score","max_score","assessments_submitted","average_submission_date","average_assessment_weight","assessment_type_diversity",
        "total_resource_clicks","average_resource_clicks","first_used","last_used","resources_used","interaction_duration","days_registered"
        ]
    # categorical features will include bool because its either 0/1
    categorical_features =[
        "gender","region","highest_education","imd_band","age_band", "disability",
    ]

    # fixed, removed all other text data
    textual_feature = "resource_text"


    # structure of code inspired by: https://stackoverflow.com/questions/69802958/how-do-i-turn-preprocessed-data-from-pipelines-into-dataframes

    # create preprocessing pipelines for each feature type

    #1 according to design specs, will fill with median for missing (if present) and scale normally using standard scaling (z score normalization)
    numerical_transformer = Pipeline(steps=[
        ('missing',SimpleImputer(strategy='median')),
        ('scaling',StandardScaler())
    ])

    #2 as per design, fill with mode for missing and one hot encode to turn its numeric (with unknowns ignored to prevent crashes)
    categorical_transformer = Pipeline(steps=[
        ('missing',SimpleImputer(strategy='most_frequent')),
        ('encoding',OneHotEncoder(handle_unknown='ignore'))
    ])
    # end inspired code


    #for text, will include directly in final preprocessor because it is one operation of TF-IDF vectorization
    preprocessor = ColumnTransformer([("numerical", numerical_transformer, numerical_features),
                                      ("categorical",categorical_transformer,categorical_features),
                                      ("textual",TfidfVectorizer(),textual_feature)])

    # now build model according to parameter, ALL models have a random state to keep same results each time + balanced class weight to account for the imbalance in interactions in dataset, setting verbose to track progress
    # first random forest
    if(model_type == "random_forest"):

        # using all CPU cores (for fast execution), initial 200 decision trees and max depth of 10 as good starting point
        # parameters code inspired by https://www.kaggle.com/code/zincbottom/oulad-random-forest#Modeling
        model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced',random_state=10,n_jobs=-1, verbose=1)
        # end inspired code

    # otherwise SVC
    elif(model_type == 'SVC'):
        # probability set to true to return float between 0 and 1 for the relevance score rather than binary
        model = SVC(class_weight='balanced',random_state=10, verbose=True)

    else:
        raise ValueError("Model type must be either random_forest or SVC")

    # final model with preprocessor as first step
    final_model = Pipeline(steps=[
                           ('preprocessing',preprocessor),
                           ('model',model)])

    # fit it with training data so it is ready to evaluate next
    final_model.fit(X_training_data,y_training_data)

    return final_model

# code inspired by: https://www.analyticsvidhya.com/blog/2023/02/how-to-save-and-load-machine-learning-models-in-python-using-joblib-library/
def save_model(model,filepath):
    joblib.dump(model, filepath)

    print("Model saved")

def load_model(filepath):
    return joblib.load(filepath)

# end inspired code

# preidction differs based on model
def predict_ML_scores(model, X_test_data, model_type):

    # note that scales are different, but does not matter as we are checking ML metrics
    if model_type == 'random_forest':
        # use probabilities
        scores = model.predict_proba(X_test_data)[:,1]

    elif model_type == "SVC": 
        # use decision function
        scores = model.decision_function(X_test_data)

    else:
        raise ValueError("Model type must be random_forest or SVC")

    return scores