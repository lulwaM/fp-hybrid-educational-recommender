#INITIAL, pasted from prototype. will adjust to work with real dataset

# for data storage
import pandas as pd

# recommender scores 1: popularity based
# input: none (becuase it is general for all students, not specific to student), output: dataframe with columns=id_site and popularity_scores
def popularity_scores(interactions):

    # creates new series where index=id site and value=sum clicks total (add up clicks for each resource)
    # note: mean is not used like in other projects with ratings because feature is interactions not range 1-5
    scores = interactions.groupby("id_site")["sum_click"].sum()

    # organize series and rename column to match score being calculated
    # note: reset index moves the id_site to normal column instead of index
    scores = scores.reset_index().rename(columns={"sum_click": "popularity_score"})

    # normalize score between 0-1 by dividing by max (makes score more understandable and comparable on same scale)
    scores["popularity_score"] = (
        scores["popularity_score"] / scores["popularity_score"].max()
    )

    return scores