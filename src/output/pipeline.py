from src.data_handling import preprocessing
from src.recommenders import popularity
from src.recommenders import collaborative
from src.recommenders import content_based
from src.output import evaluation
from src.data_handling import exploration

# TESTING THAT BASIC RECOMMENDERS WORK INDEPENDENTLY

#first load/merge/clean/typecast/process final data

loaded_datasets = preprocessing.load_datasets()
merged_datasets = preprocessing.merge_datasets(loaded_datasets)
cleaned_datasets = preprocessing.clean_datasets(merged_datasets)
typecast_datasets = preprocessing.typecasting_datasets(cleaned_datasets)

processed_datasets = preprocessing.processing_datasets(typecast_datasets)

evaluated_student = 11391

def check_baselines_working():
    #recommender 1: popularity
    popularity_recs = popularity.popularity_scores(evaluated_student,processed_datasets["interaction_data"])

    print("Popularity Recommender: \n")
    print(popularity_recs.head(5))

    # next is collaborative recommender, for student id 11391
    collaborative_recs = collaborative.collaborative_scores(evaluated_student,processed_datasets["interaction_data"])
    print("\nCollaborative Filtering Recommender: \n")
    print(collaborative_recs.head(5))

    #last is content based recommender, for student id 11391
    content_recs = content_based.content_scores(evaluated_student,processed_datasets["resource_data"],processed_datasets["interaction_data"])
    print("\nContent-based Filtering Recommender: \n")
    print(content_recs.head(5))

# check_baselines_working()

#BASELINE RECOMMENDERS EVALUATION for top 20 recommendation
# history_interactions, future_interactions = evaluation.temporal_split(processed_datasets["vle_data"],86)
# k = 20

# evaluation.evaluate_baselines_performances(history_interactions,future_interactions,processed_datasets["resource_data"],evaluated_student,k)

# results, summary = evaluation.evaluate_baseline_models(history_interactions,future_interactions,processed_datasets["resource_data"],k)
# print(summary)

#save visualizations in images folder
exploration.explore_student_data(processed_datasets["student_data"])
exploration.explore_assessment_data(processed_datasets["assessment_data"])
exploration.explore_vle_data(processed_datasets["vle_data"])
exploration.explore_resource_data(processed_datasets["resource_data"])
exploration.explore_interaction_data(processed_datasets["interaction_data"])
