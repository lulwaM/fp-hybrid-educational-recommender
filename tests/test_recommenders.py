# standard imports for python testing framework, data storage, and numerical calculations
import unittest
import pandas as pd
import numpy as np

# to create sample test data and preprocessed
from tests.sample_data import create_datasets
from src.data_handling import preprocessing, feature_engineering
from src.output import evaluation

# functions to be tested
from src.recommenders import popularity, collaborative, content_based, hybrid, ranking


# test suite for model files within recommenders folder
# input: testcase from unittest library (required for testing) / no output, simply calls test cases for unit testing
class TestRecommenders(unittest.TestCase):

    # runs once before tests, used to store datasets to be evaluated (prevents code repetition)
    def setUp(self):
        # perform entire preprocessing pipeline on sample data
        self.datasets = create_datasets()
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)

        # final preprocessed dataset
        self.processed_datasets = preprocessing.processing_datasets(typecast_result)

    # 1. popularity recommender scores function

    # test case for checking correct value for popularity model
    def test_popularity_value(self):
        # since 101 id site material is most clicked on, it should be first recommendation with no records
        popularity_scores = popularity.popularity_scores(
            4, self.processed_datasets["interaction_data"]
        )

        # check expected value of first record
        self.assertEqual(101, popularity_scores.iloc[0]["id_site"])

    # test case to check that a dataframe is returned
    def test_popularity_dataframe(self):
        # on id student 1
        popularity_scores = popularity.popularity_scores(
            1, self.processed_datasets["interaction_data"]
        )

        # scores should be returned as dataframe
        self.assertIsInstance(popularity_scores, pd.DataFrame)

    # test case for cold start handling by popularity model
    def test_popularity_cold_start(self):
        # on id student 4 which does not exist in sample data
        popularity_scores = popularity.popularity_scores(
            4, self.processed_datasets["interaction_data"]
        )

        # returns recommendations from ALL courses
        self.assertIn("AAA", popularity_scores["code_module"].values)
        self.assertIn("BBB", popularity_scores["code_module"].values)

    # test case for popularity model scores within interacted with courses
    def test_popularity_taken_course(self):
        # student id 1 only takes course with code module AAA, not BBB, so recommendations only from there
        popularity_scores = popularity.popularity_scores(
            1, self.processed_datasets["interaction_data"]
        )

        # only restricted courses recommended
        self.assertIn("AAA", popularity_scores["code_module"].values)
        self.assertNotIn("BBB", popularity_scores["code_module"].values)

    # 2. collaborative recommender scores function

    # test case for checking correct value for collaborative filtering model
    def test_collaborative_value(self):
        # student id 3 already used resource id 101 and 103, most similar to student id 2 which used resource id 102 (so should be top recommendation)
        collaborative_scores = collaborative.collaborative_scores(
            3, self.processed_datasets["interaction_data"]
        )

        # check expceted value of first record
        self.assertEqual(102, collaborative_scores.iloc[0]["id_site"])

    # test case to check that used resources are excluded from recommendations
    def test_collaborative_used_resources(self):
        # student id 1 used resources with id 101 and 102, so should not exist
        collaborative_scores = collaborative.collaborative_scores(
            1, self.processed_datasets["interaction_data"]
        )

        # used resource ids not in recommendations
        self.assertNotIn(101, collaborative_scores["id_site"].values)
        self.assertNotIn(102, collaborative_scores["id_site"].values)

    # test case for collaborative model recommender expected columns
    def test_collaborative_columns(self):
        # perform function to be tested
        collaborative_scores = collaborative.collaborative_scores(
            1, self.processed_datasets["interaction_data"]
        )

        # columns should contain resource details + score (must follow same order else test fails)
        self.assertEqual(
            collaborative_scores.columns.to_list(),
            [
                "id_site",
                "code_module",
                "code_presentation",
                "activity_type",
                "collaborative_score",
            ],
        )

    # test case for checking resource data exists in collaborative recommendations
    def test_collaborative_resource_data(self):
        # perform function to be tested
        collaborative_scores = collaborative.collaborative_scores(
            1, self.processed_datasets["interaction_data"]
        )

        # no missing values for resource info
        self.assertFalse(collaborative_scores["code_module"].isna().any())
        self.assertFalse(collaborative_scores["code_presentation"].isna().any())
        self.assertFalse(collaborative_scores["activity_type"].isna().any())

    # 3. content-based recommender scores function

    # test case for checking correct value for content based filtering model
    def test_content_value(self):
        # student id 1 already used resource 101 and 102, should recommender resoruce 103 because it is closest to used resource 103 (both url activity type)
        content_scores = content_based.content_scores(
            1,
            self.processed_datasets["resource_data"],
            self.processed_datasets["interaction_data"],
        )

        # check expected value of first resource
        self.assertEqual(103, content_scores.iloc[0]["id_site"])

    # test case for checking content scores within expected normalized range
    def test_content_normalized(self):
        # perform function to be tested
        content_scores = content_based.content_scores(
            1,
            self.processed_datasets["resource_data"],
            self.processed_datasets["interaction_data"],
        )

        # score values between 0 and 1
        self.assertGreaterEqual(1, content_scores["content_score"].values)
        self.assertLessEqual(0, content_scores["content_score"].values)

    # test case for checking scores are sorted in descending order (highest relevancy first)
    def test_content_sorted(self):
        # perform function to be tested
        content_scores = content_based.content_scores(
            1,
            self.processed_datasets["resource_data"],
            self.processed_datasets["interaction_data"],
        )

        # checks if sorted, code copied from: https://stackoverflow.com/questions/17315881/how-can-i-check-if-a-pandas-dataframes-index-is-sorted
        self.assertTrue(content_scores["content_score"].is_monotonic_decreasing)
        # end copied code

    # test case for no recommendations case for content based filtering model
    def test_content_empty(self):
        # student id 2 only used resource ID 201 from course BBB which has only this single resource, so no recommendations
        content_scores = content_based.content_scores(
            2,
            self.processed_datasets["resource_data"],
            self.processed_datasets["interaction_data"],
        )

        # dataframe should be empty for this student
        self.assertTrue(content_scores.empty)

    # 4. hybrid recommender function (fixed weight)

    # test case for checking correct value for hybrid model
    def test_hybrid_value(self):
        # creating new scores to calculate according to hybrid fixed weightage formula, using single recommendation for sample only
        popularity_score = pd.DataFrame(
            {
                "id_site": [101],
                "code_module": ["AAA"],
                "code_presentation": ["2013J"],
                "activity_type": ["url"],
                "popularity_score": [0.4],
            }
        )

        content_score = pd.DataFrame(
            {
                "id_site": [101],
                "code_module": ["AAA"],
                "code_presentation": ["2013J"],
                "activity_type": ["url"],
                "content_score": [0.5],
            }
        )

        collaborative_score = pd.DataFrame(
            {
                "id_site": [101],
                "code_module": ["AAA"],
                "code_presentation": ["2013J"],
                "activity_type": ["url"],
                "collaborative_score": [0.3],
            }
        )

        # perform function to be tested
        hybrid_score = hybrid.hybrid_scores(
            popularity_score, content_score, collaborative_score
        )

        # using new sample scores, perform formula according to weightage
        expected_score = (
            (popularity_score.loc[0, "popularity_score"] * 0.4)
            + (content_score.loc[0, "content_score"] * 0.2)
            + (collaborative_score.loc[0, "collaborative_score"] * 0.4)
        )

        # calculated according to default weights, check if expected
        self.assertEqual(hybrid_score.loc[0, "hybrid_score"], expected_score)

    # test case for some scores being 0 instead of NaN when not recommended by all models in hybrid
    def test_hybrid_missing(self):

        # creating new scores to calculate according to hybrid fixed weightage formula, using single recommendation for sample only
        popularity_score = pd.DataFrame(
            {
                "id_site": [101],
                "code_module": ["AAA"],
                "code_presentation": ["2013J"],
                "activity_type": ["url"],
                "popularity_score": [0.4],
            }
        )

        # content only recommends 102, the rest recommend 101 (will be missing scores)
        content_score = pd.DataFrame(
            {
                "id_site": [102],
                "code_module": ["AAA"],
                "code_presentation": ["2013J"],
                "activity_type": ["url"],
                "content_score": [0.5],
            }
        )

        # same idea for collaborative model
        collaborative_score = pd.DataFrame(
            {
                "id_site": [101],
                "code_module": ["AAA"],
                "code_presentation": ["2013J"],
                "activity_type": ["url"],
                "collaborative_score": [0.3],
            }
        )

        # performs function ot be tested
        hybrid_score = hybrid.hybrid_scores(
            popularity_score, content_score, collaborative_score
        )

        # works by extracting hybrid record for a specific resource id, then locating the first entry for each score via iloc

        # for 101, will be 0 score for content (not recommended)
        self.assertEqual(
            hybrid_score[hybrid_score["id_site"] == 101].iloc[0]["content_score"], 0
        )

        # for 102, will be 0 score for both populatity and collaborative (not recommended)
        self.assertEqual(
            hybrid_score[hybrid_score["id_site"] == 102].iloc[0]["collaborative_score"],
            0,
        )
        self.assertEqual(
            hybrid_score[hybrid_score["id_site"] == 102].iloc[0]["popularity_score"], 0
        )

    # test case for raising error when weights do not add up to 1
    def test_hybrid_error_greater(self):
        # creating new scores to calculate according to hybrid fixed weightage formula, using single recommendation for sample only
        popularity_score = pd.DataFrame(
            {
                "id_site": [101],
                "code_module": ["AAA"],
                "code_presentation": ["2013J"],
                "activity_type": ["url"],
                "popularity_score": [0.4],
            }
        )

        content_score = pd.DataFrame(
            {
                "id_site": [101],
                "code_module": ["AAA"],
                "code_presentation": ["2013J"],
                "activity_type": ["url"],
                "content_score": [0.5],
            }
        )

        collaborative_score = pd.DataFrame(
            {
                "id_site": [101],
                "code_module": ["AAA"],
                "code_presentation": ["2013J"],
                "activity_type": ["url"],
                "collaborative_score": [0.3],
            }
        )

        # if weights sum > 1, error returned
        # code copied from: https://stackoverflow.com/questions/129507/how-do-you-test-that-a-python-function-throws-an-exception
        with self.assertRaises(ValueError) as context:
            hybrid.hybrid_scores(
                popularity_score, content_score, collaborative_score, 1, 2, 3
            )
        # end copied code

    # test case for raising error when weights are negative
    def test_hybrid_error_negative(self):
        # creating new scores to calculate according to hybrid fixed weightage formula, using single recommendation for sample only
        popularity_score = pd.DataFrame(
            {
                "id_site": [101],
                "code_module": ["AAA"],
                "code_presentation": ["2013J"],
                "activity_type": ["url"],
                "popularity_score": [0.4],
            }
        )

        content_score = pd.DataFrame(
            {
                "id_site": [101],
                "code_module": ["AAA"],
                "code_presentation": ["2013J"],
                "activity_type": ["url"],
                "content_score": [0.5],
            }
        )

        collaborative_score = pd.DataFrame(
            {
                "id_site": [101],
                "code_module": ["AAA"],
                "code_presentation": ["2013J"],
                "activity_type": ["url"],
                "collaborative_score": [0.3],
            }
        )

        # if any weight <0, error returned
        # code copied from: https://stackoverflow.com/questions/129507/how-do-you-test-that-a-python-function-throws-an-exception
        with self.assertRaises(ValueError) as context:
            hybrid.hybrid_scores(
                popularity_score, content_score, collaborative_score, -0.3, 0.5, 0.8
            )
        # end copied code

    # now for ranking file functions

    # 5. generate candidates function

    # test case for candidate recommendations having all scores
    def test_generate_candidates_columns(self):

        # call candidate recommendation on student id 1, who has interactions with course AAA presentation 2013J
        candidates_result = ranking.generate_candidates(
            id_student=1,
            interaction_data=self.processed_datasets["interaction_data"],
            resource_data=self.processed_datasets["resource_data"],
            code_module="AAA",
            code_presentation="2013J",
            k=20,
        )

        # check scores generated from all recommenders
        self.assertIn("popularity_score", candidates_result.columns)
        self.assertIn("collaborative_score", candidates_result.columns)
        self.assertIn("content_score", candidates_result.columns)

    # test case for no scores being missing because filled with 0
    def test_generate_candidates_missing(self):

        # call candidate recommendation on student id 1, who has interactions with course AAA presentation 2013J
        candidates_result = ranking.generate_candidates(
            id_student=1,
            interaction_data=self.processed_datasets["interaction_data"],
            resource_data=self.processed_datasets["resource_data"],
            code_module="AAA",
            code_presentation="2013J",
            k=20,
        )

        # any missing scores must be filled with 0, so no NaN values
        self.assertFalse(candidates_result["popularity_score"].isna().any())
        self.assertFalse(candidates_result["collaborative_score"].isna().any())
        self.assertFalse(candidates_result["content_score"].isna().any())

    # 6. create ML dataset function

    # test case for features created in ML dataset
    def test_ml_dataset_columns(self):

        # creating simple temporal split to produce history/future interactions for create ml dataset function, choosing cutoff = 1 because it exists for sample data
        history_interactions, future_interactions = evaluation.temporal_split(
            vle_data=self.processed_datasets["vle_data"], cutoff_day=1
        )

        # also creating features for function, same cutoff day of 1
        student_features = feature_engineering.create_features(
            datasets=self.processed_datasets, cutoff_day=1
        )

        # create ml dataset
        ml_dataset_result = ranking.create_ml_dataset(
            history_interactions=history_interactions,
            future_interactions=future_interactions,
            resource_data=self.processed_datasets["resource_data"],
            student_features=student_features,
            k=20,
        )

        # ML target created (most important)
        self.assertIn("target", ml_dataset_result.columns)

        # also student features
        self.assertIn("days_registered", ml_dataset_result.columns)
        self.assertIn("average_score", ml_dataset_result.columns)
        self.assertIn("total_resource_clicks", ml_dataset_result.columns)

    # test case for checking value of target column in ML dataset
    def test_ml_dataset_target(self):

        # creating simple temporal split to produce history/future interactions for create ml dataset function, choosing cutoff = 1 because it exists for sample data
        history_interactions, future_interactions = evaluation.temporal_split(
            vle_data=self.processed_datasets["vle_data"], cutoff_day=1
        )

        # also creating features for function, same cutoff day of 1
        student_features = feature_engineering.create_features(
            datasets=self.processed_datasets, cutoff_day=1
        )

        # create ml dataset
        ml_dataset_result = ranking.create_ml_dataset(
            history_interactions=history_interactions,
            future_interactions=future_interactions,
            resource_data=self.processed_datasets["resource_data"],
            student_features=student_features,
            k=20,
        )

        # check target either 0 or 1 (binary)
        self.assertTrue(ml_dataset_result["target"].isin([0, 1]).all())

    # 7. sample training data function

    # test case to check sample size, very simple because it simply uses sklearn well-established function
    def test_sample_training_data_size(self):

        # creating simple temporal split to produce history/future interactions for create ml dataset function, choosing cutoff = 1 because it exists for sample data
        history_interactions, future_interactions = evaluation.temporal_split(
            vle_data=self.processed_datasets["vle_data"], cutoff_day=1
        )

        # also creating features for function, same cutoff day of 1
        student_features = feature_engineering.create_features(
            datasets=self.processed_datasets, cutoff_day=1
        )

        # create ml dataset to SAMPLE
        ml_dataset = ranking.create_ml_dataset(
            history_interactions=history_interactions,
            future_interactions=future_interactions,
            resource_data=self.processed_datasets["resource_data"],
            student_features=student_features,
            k=20,
        )

        # take 2 sample rows for training data (note that 2 chosen because there are low number of interactions in sample data)
        sample_result = ranking.sample_training_data(
            training_dataset=ml_dataset, sample_size=2
        )

        # check if actually 2 rows taken
        self.assertEqual(len(sample_result), 2)

    # 8. prepare model input function

    # test case for checking X features of ML dataset
    def test_model_input_columns(self):

        # creating simple temporal split to produce history/future interactions for create ml dataset function, choosing cutoff = 1 because it exists for sample data
        history_interactions, future_interactions = evaluation.temporal_split(
            vle_data=self.processed_datasets["vle_data"], cutoff_day=1
        )

        # also creating features for function, same cutoff day of 1
        student_features = feature_engineering.create_features(
            datasets=self.processed_datasets, cutoff_day=1
        )

        # create ml dataset to PREPARE MODEL INPUT
        ml_dataset = ranking.create_ml_dataset(
            history_interactions=history_interactions,
            future_interactions=future_interactions,
            resource_data=self.processed_datasets["resource_data"],
            student_features=student_features,
            k=20,
        )

        # NOTE: supplying same dataset for training and test because this test does not evaluate uniqueness, simply evaluates structure of data returned
        X_train, y_train, X_test, y_test = ranking.prepare_model_input(
            training_dataset=ml_dataset, test_dataset=ml_dataset
        )

        # identifiers removed from model input (X for features, both train/test)
        self.assertNotIn("id_student", X_train.columns)
        self.assertNotIn("id_site", X_train.columns)
        self.assertNotIn("id_student", X_test.columns)
        self.assertNotIn("id_site", X_test.columns)

        # useful features should exist
        self.assertIn("gender", X_train.columns)
        self.assertIn("region", X_train.columns)
        self.assertIn("average_score", X_test.columns)
        self.assertIn("popularity_score", X_test.columns)

    # test case to check X in ML dataset data types
    def test_model_input_type(self):

        # creating simple temporal split to produce history/future interactions for create ml dataset function, choosing cutoff = 1 because it exists for sample data
        history_interactions, future_interactions = evaluation.temporal_split(
            vle_data=self.processed_datasets["vle_data"], cutoff_day=1
        )

        # also creating features for function, same cutoff day of 1
        student_features = feature_engineering.create_features(
            datasets=self.processed_datasets, cutoff_day=1
        )

        # create ml dataset to PREPARE MODEL INPUT
        ml_dataset = ranking.create_ml_dataset(
            history_interactions=history_interactions,
            future_interactions=future_interactions,
            resource_data=self.processed_datasets["resource_data"],
            student_features=student_features,
            k=20,
        )

        # NOTE: supplying same dataset for training and test because this test does not evaluate uniqueness, simply evaluates structure of data returned
        X_train, y_train, X_test, y_test = ranking.prepare_model_input(
            training_dataset=ml_dataset, test_dataset=ml_dataset
        )

        # check categorical columns are objects
        self.assertEqual(X_train["gender"].dtype, object)
        self.assertEqual(X_train["region"].dtype, object)
        self.assertEqual(X_test["highest_education"].dtype, object)
        self.assertEqual(X_test["disability"].dtype, object)

    # 9. create ml model function

    # test case to raise error if model type not supported, very simple because it simply uses sklearn well-established function
    def test_create_model_error(self):

        # if random_forest or SVC not model_type, error returned
        # code copied from: https://stackoverflow.com/questions/129507/how-do-you-test-that-a-python-function-throws-an-exception
        with self.assertRaises(ValueError) as context:
            ranking.create_ML_model(None, None, model_type="error")
        # end copied code

    # 10. predict ml scores function

    # test case to raise error if model type not supported, very simple because it simply uses sklearn well-established function
    def test_predict_scores_error(self):

        # if random_forest or SVC not model_type, error returned
        # code copied from: https://stackoverflow.com/questions/129507/how-do-you-test-that-a-python-function-throws-an-exception
        with self.assertRaises(ValueError) as context:
            ranking.predict_ML_scores(None, None, model_type="error")
        # end copied code

    # 11. prepare prediction input function

    # test case to check no missing values in column as filled with empty string
    def test_prediction_input_missing(self):

        # creating simple temporal split to produce history/future interactions for create ml dataset function, choosing cutoff = 1 because it exists for sample data
        history_interactions, future_interactions = evaluation.temporal_split(
            vle_data=self.processed_datasets["vle_data"], cutoff_day=1
        )

        # also creating features for function, same cutoff day of 1
        student_features = feature_engineering.create_features(
            datasets=self.processed_datasets, cutoff_day=1
        )

        # create ml dataset to PREPARE MODEL INPUT
        ml_dataset = ranking.create_ml_dataset(
            history_interactions=history_interactions,
            future_interactions=future_interactions,
            resource_data=self.processed_datasets["resource_data"],
            student_features=student_features,
            k=20,
        )

        # simulate missing resource text, should be filled with empty string
        ml_dataset.loc[0, "resource_text"] = np.nan

        # perform function to be tested
        X = ranking.prepare_prediction_input(ML_dataset=ml_dataset)

        # X should fill the NaN with empty string
        self.assertEqual(X.loc[0, "resource_text"], "")

    # test case for checking train/test X data have same exact columns to feed model
    def test_prediction_input_features(self):

        # creating simple temporal split to produce history/future interactions for create ml dataset function, choosing cutoff = 1 because it exists for sample data
        history_interactions, future_interactions = evaluation.temporal_split(
            vle_data=self.processed_datasets["vle_data"], cutoff_day=1
        )

        # also creating features for function, same cutoff day of 1
        student_features = feature_engineering.create_features(
            datasets=self.processed_datasets, cutoff_day=1
        )

        # create ml dataset to PREPARE MODEL INPUT
        ml_dataset = ranking.create_ml_dataset(
            history_interactions=history_interactions,
            future_interactions=future_interactions,
            resource_data=self.processed_datasets["resource_data"],
            student_features=student_features,
            k=20,
        )

        # both the X from prepare model input and prepare prediction input should have exact same features
        X_prediction = ranking.prepare_prediction_input(ML_dataset=ml_dataset)
        # NOTE: supplying same dataset for training and test because this test does not evaluate uniqueness, simply evaluates structure of data returned
        X_train, y_train, X_test, y_test = ranking.prepare_model_input(
            training_dataset=ml_dataset, test_dataset=ml_dataset
        )

        # check equal columns from both
        self.assertEqual(X_prediction.columns.to_list(), X_train.columns.to_list())

    # NOT testing ranking saving/loading model because they are from well-established joblib
