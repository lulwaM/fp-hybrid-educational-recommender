#for data storage
import pandas as pd

#creates model interpretability, note that recommendation is a record with all the scores attached, using fixed weight hybrid weights as default
def explain_recommendation(recommendation,popularity_weight=0.4,content_weight=0.2,collaborative_weight=0.4):

    weighted_scores = {"popularity": recommendation["popularity_score"] * popularity_weight,
              "content":recommendation["content_score"] * content_weight,
               "collaborative": recommendation["collaborative_score"] * collaborative_weight}

    #code copied from: https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
    highest_score = max(weighted_scores, key=weighted_scores.get)
    #end copied code

    explanation = ""

    if highest_score == "popularity":
        explanation = "This resource was recommended because it is frequently used by other students"

    if highest_score == "content":
        explanation = "This resource was recommended because it is similar to other interacted-with resources"

    if highest_score == "collaborative":
        explanation = "This resource was recommended because it is used by similar students"

    return explanation