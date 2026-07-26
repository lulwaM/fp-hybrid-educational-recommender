from src.data_handling import preprocessing
from src.recommenders import popularity
from src.recommenders import collaborative
from src.recommenders import content_based

# TESTING THAT BASIC RECOMMENDERS WORK INDEPENDENTLY

#first load/merge/clean/typecast/process final data

loaded_datasets = preprocessing.load_datasets()
merged_datasets = preprocessing.merge_datasets(loaded_datasets)
cleaned_datasets = preprocessing.clean_datasets(merged_datasets)
typecast_datasets = preprocessing.typecasting_datasets(cleaned_datasets)

processed_datasets = preprocessing.processing_datasets(typecast_datasets)

#recommender 1: popularity
popularity_scores = popularity.popularity_scores(processed_datasets["interaction_data"])

print("Popularity Recommender: \n")
print(popularity_scores.head(5))

# next is collaborative recommender, for student id 11391
collaborative_scores = collaborative.collaborative_scores(11391,processed_datasets["interaction_data"])
print("\nCollaborative Filtering Recommender: \n")
print(collaborative_scores.head(5))

#last is content based recommender, for student id 11391
content_scores = content_based.content_scores(11391,processed_datasets["resource_data"],processed_datasets["interaction_data"])
print("\nContent-based Filtering Recommender: \n")
print(content_scores.head(5))

# next steps:
# add cutoff logic in evaluation file,
# create evaluation functions precision/accuracy/recall,
# evaluate each recommenders' performance

