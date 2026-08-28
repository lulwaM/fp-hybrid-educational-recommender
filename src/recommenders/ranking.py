#imports
import pandas as pd

# ML ranking model will use ALL recommenders
from src.recommenders import popularity, collaborative, content_based

# generate possible recommendations for a student-course record, k = number of candidate recommendations to generate
def generate_candidates(id_student,interaction_data,resource_data, k=20, popularity_recs=None, collaborative_recs=None, content_recs=None):

    if(popularity_recs is None or collaborative_recs is None or content_recs is None ):
        # generate scores since not generated

        popularity_recs = popularity.popularity_scores(id_student,interaction_data)
        collaborative_recs = collaborative.collaborative_scores(id_student,interaction_data)
        content_recs = content_based.content_scores(id_student,resource_data,interaction_data)


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
        candidates = generate_candidates(evaluated_student["id_student"],history_interactions,resource_data,k)

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

    # return train/test X/y
    return (X_train, y_train, X_test, y_test)


