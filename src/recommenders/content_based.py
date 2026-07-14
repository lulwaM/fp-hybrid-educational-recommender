# for data storage
import pandas as pd

# for collaborative and content based filtering
from sklearn.metrics.pairwise import cosine_similarity

# for content based filtering
from sklearn.feature_extraction.text import TfidfVectorizer

# recommender scores 3: content based filtering
# input: student id to make targetted recommendations, interactions df to see similar students, and resources df to view similar resources, output: dataframe with columns=resource id and content based filtering score
def content_scores(id_student,resources, interactions):

    # text conversion code inspiration from: https://medium.com/@sumanadhikari/building-a-movie-recommendation-engine-using-scikit-learn-8dbb11c5aa4b

    # store course text features in one to be vectorized (add space to maintain word meaning)
    resources["course_text"] = (
        resources["title"]
        + " "
        + resources["code_module"]
        + " "
        + resources["code_presentation"]
        + " "
        + resources["activity_type"]
    )

    # get numerical represetnation of course text, returns matrix
    # code syntax/workflow from scikit learn documentation
    vectorizer = TfidfVectorizer()
    resource_matrix = vectorizer.fit_transform(resources["course_text"])

    # end inspired code

    # use cosine similarity, but on resources this time not students
    # code syntax usage from sci kit learn documentation
    content_similarity = cosine_similarity(resource_matrix)

    # as mentioned previously, returns matrix so convert to dataframe for easier access (both index and columns are resource id because we are comparing resources only)
    content_similarity_df = pd.DataFrame(
        content_similarity, index=resources["id_site"], columns=resources["id_site"]
    )

    # obtain used resources row of data from student id filtering, only select the resource id column UNIQUE values (could have multiple interactions with same resource)
    used_resources = interactions[interactions["id_student"] == id_student]["id_site"]
    used_resources = used_resources.unique()

    # start with empty content based scores, will populate in for loop
    scores = pd.Series(dtype=float)

    # iterate over each used resource of student
    # iteration code inspiration from: https://www.scaler.com/topics/machine-learning/content-based-filtering/
    for id_site in used_resources:

        # add used rseource similarity to all sources as series (accumulate similarity of used sources to all sources)
        if id_site in content_similarity_df.index:
            # fill value fail safe if missing similarity, output is series with each resource id and its total similarity to student's used resources
            # note: .add used instead of local vars because it is vectorized operation (faster, computationally efficient)
            scores = scores.add(content_similarity_df[id_site], fill_value=0)
    # end inspired code

    # remove already used resources to not recommend
    scores = scores.drop(used_resources)

    # organize scores into dataframe with correct format so that it has resource id and corresponding content based score
    # note: reset index moves the id_site to normal column instead of index
    scores = scores.reset_index()
    scores.columns = ["id_site", "content_score"]

    # normalize scores between 0 and 1 to be understandable, if condition prevents errors with divisons by 0 or empty
    if len(scores) > 0 and scores["content_score"].max() > 0:
        scores["content_score"] = (
            scores["content_score"] / scores["content_score"].max()
        )

    return scores