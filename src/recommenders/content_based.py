# for data storage
import pandas as pd

# for collaborative and content based filtering
from sklearn.metrics.pairwise import cosine_similarity

# for content based filtering
from sklearn.feature_extraction.text import TfidfVectorizer

# recommender scores 3: content based filtering
# input: student id to make targetted recommendations, interactions df to see similar students, and resources df to view similar resources, output: dataframe with columns=resource id and content based filtering score
def content_scores(id_student,resource_data, interaction_data):

    # text conversion code inspiration from: https://medium.com/@sumanadhikari/building-a-movie-recommendation-engine-using-scikit-learn-8dbb11c5aa4b

    # get numerical represetnation of course text, returns matrix
    # code syntax/workflow from scikit learn documentation
    vectorizer = TfidfVectorizer()
    resource_matrix = vectorizer.fit_transform(resource_data["resource_text"])

    # end inspired code

    # use cosine similarity, but on resources this time not students
    # code syntax usage from sci kit learn documentation
    content_similarity = cosine_similarity(resource_matrix)

    # as mentioned previously, returns matrix so convert to dataframe for easier access (both index and columns are resource id because we are comparing resources only)
    content_similarity_df = pd.DataFrame(
        content_similarity, index=resource_data["id_site"], columns=resource_data["id_site"]
    )

    # obtain used resources row of data from student id filtering, only select the resource id column UNIQUE values (could have multiple interactions with same resource)
    used_resources = interaction_data[interaction_data["id_student"] == id_student]["id_site"]
    used_resources = used_resources.unique()

    # start with empty content based scores, will populate in for loop
    scores = pd.Series(dtype=float)

    #cold start mitigation, user has no interactions
    if(len(used_resources) == 0):
        # empty dataframe, no content based score
        return pd.DataFrame(columns=["id_site","code_module","code_presentation","activity_type","content_score"])

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

    # include resource details like code module/presentation in recommender function
    #first get list of resource details with no duplicates
    resource_details = interaction_data[["id_site","code_module","code_presentation","activity_type"]].drop_duplicates()

    #merge these details with the final resource scores
    scores = scores.merge(resource_details, on="id_site",how="left")

    #display highest score first, drop old indices for cleaner display
    scores = scores.sort_values(by="content_score", ascending=False).reset_index(drop=True)

    #re-organize columns to be consistent with other recommender outputs
    scores = scores[["id_site","code_module","code_presentation","activity_type","content_score"]]

    return scores