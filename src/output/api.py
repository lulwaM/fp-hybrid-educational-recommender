# standard import for API setup and JSON response formatting
from fastapi import FastAPI, HTTPException
import json

# import for loading API datasets
from src.data_handling import preprocessing

# import for loading final ML random forest ranking model
from src.recommenders import ranking

# import for main API recommendation generation function
from src.output import pipeline

# creating API and setting key identifiers
app = FastAPI(
    title="Educational Content Recommender API",
    description="Returns personalized educational recommendations based on id_student, code_module, code_presentation inputs",
)


# first, do actions before any route to do it once (not re-load each time)

# load API dataset and model to be used in generation
datasets = preprocessing.load_api_datasets()
model = ranking.load_model("models/random_forest_model.joblib")

# return top 20 recs by default
k = 20


# initial get route code copied and adapted from: https://fastapi.tiangolo.com/tutorial/first-steps/
@app.get(
    "/",
    summary="Entry page",
    description="Provides instructions on accessing app routes to make a request",
)
# simply returns JSON format instructions on generating recommendations
async def root():
    return {
        "message": "Educational content recommender API is running! Please visit the /docs page to view routes and make a request."
    }


# end copied and adapted code


# route to generate recs, use GET HTTP request (as per conventions)
# using id_student in path parameter with the rest of identifiers in query, adding summary/header for clearer purpose in docs page
@app.get(
    "/recommendations/{id_student}",
    summary="Generate student recommendations",
    description="Returns the top 20 educational resource recommendations for a specified student id, code module, and code presentation, using 86 as the cutoff day",
)
# must define data type of inputs, which are the identifiers for recommendation generation
def get_recommendations(id_student: int, code_module: str, code_presentation: str):
    # end inspired code

    # wrap in try/except for error handling
    try:

        # call function to generate recommendaitons using identifiers / loaded datasets / loaded model / k value
        recommendations = pipeline.generate_API_recommendations(
            id_student=id_student,
            code_module=code_module,
            code_presentation=code_presentation,
            datasets=datasets,
            model=model,
            k=k,
        )

        # converting from dataframe to json to Python object to display correctly, code inspired by: https://stackoverflow.com/questions/71203579/how-to-return-a-csv-file-pandas-dataframe-in-json-format-using-fastapi
        res = recommendations.to_json(orient="records")
        parsed = json.loads(res)
        return parsed
    # end inspired code

    except ValueError as error:
        #    raising error, code copied and adapted from: https://fastapi.tiangolo.com/tutorial/handling-errors/#raise-an-httpexception-in-your-code
        raise HTTPException(status_code=404, detail=str(error))
    # end copied and adapted code
