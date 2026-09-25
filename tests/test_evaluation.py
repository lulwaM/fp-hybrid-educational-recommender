# standard imports for python testing framework, data storage, and numerical calculations
import unittest
import pandas as pd
import numpy as np

# import to create sample test data and preprocess it
from tests.sample_data import create_datasets
from src.data_handling import preprocessing

# function to be tested
from src.output import evaluation


# test suite for evaluation file
# input: testcase from unittest library (required for testing) / no output, simply calls test cases for unit testing
class TestEvaluation(unittest.TestCase):

    # runs once before tests, used to store datasets to be evaluated (prevents code repetition)
    def setUp(self):
        # perform entire preprocessing pipeline on sample data
        self.datasets = create_datasets()
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)

        # final preprocessed dataset
        self.processed_datasets = preprocessing.processing_datasets(typecast_result)

    # 1. temporal split function

    # test case for checking interactions filtered according to cutoff date
    def test_temporal_split_cutoff(self):
        # setting cutoff day at 22 (as per sample data) so that history interactions less than that and future interactions more than that + both exist
        history_interactions, future_interactions = evaluation.temporal_split(
            self.processed_datasets["vle_data"], 22
        )

        # note that date split into 2 new variables (last/first used), each correspond to cutoff date
        self.assertTrue((history_interactions["last_used"] <= 22).all())
        self.assertTrue((future_interactions["first_used"] > 22).all())

    # test case for checking features in temporal split
    def test_temporal_split_columns(self):
        # setting cutoff day at 22 (as per sample data) so that history interactions less than that and future interactions more than that + both exist
        history_interactions, future_interactions = evaluation.temporal_split(
            self.processed_datasets["vle_data"], 22
        )

        # all new variables created via aggregation + existing student/resource info (note that the order remains same as creation to prevent test fail)
        self.assertEqual(
            [
                "id_student",
                "code_module",
                "code_presentation",
                "id_site",
                "activity_type",
                "total_resource_clicks",
                "first_used",
                "last_used",
                "click_duration",
            ],
            history_interactions.columns.to_list(),
        )
        self.assertEqual(
            [
                "id_student",
                "code_module",
                "code_presentation",
                "id_site",
                "activity_type",
                "total_resource_clicks",
                "first_used",
                "last_used",
                "click_duration",
            ],
            future_interactions.columns.to_list(),
        )

    # 2. precision@k function

    # test case for checking correct result of precision evaluation metric
    def test_precision_value(self):
        # not using sample data, creating own data for clarity on precision formula
        relevant = [1, 5, 2]
        recommended = [1, 4, 2, 3]

        # precision = number of relevant items / total number of items = 2 / 4
        result = evaluation.precision_k(relevant, recommended, 4)

        # check correct value according to input
        self.assertEqual(result, 2 / 4)

    # test case for checking value is 0 when relevant items empty
    def test_precision_error_relevant(self):
        # not using sample data, creating own data for clarity on precision formula
        relevant = []
        recommended = [1, 4, 2, 3]

        # if relevant is empty, then 0 returned according to formula
        result = evaluation.precision_k(relevant, recommended, 4)

        # check correct value according to input
        self.assertEqual(0, result)

    # 3. recall@k function

    # test case for checking correct result of recall evaluation metric
    def test_recall_value(self):
        # not using sample data, creating own data for clarity on recall formula
        relevant = [1, 5, 2]
        recommended = [1, 4, 2, 3]

        # recall = number of relevant items / total number of relevant items = 2 / 3
        result = evaluation.recall_k(relevant, recommended, 4)

        # check correct value according to input
        self.assertEqual(result, 2 / 3)

    # test case for checking error is raised when denominator is 0
    def test_recall_error_k(self):
        # not using sample data, creating own data for clarity on recall formula
        relevant = [1, 5, 2]
        recommended = [1, 4, 2, 3]

        # if k < 0, then error returned
        # code copied from: https://stackoverflow.com/questions/129507/how-do-you-test-that-a-python-function-throws-an-exception
        with self.assertRaises(ValueError) as context:
            evaluation.recall_k(relevant, recommended, -1)
        # end copied code

    # 3. random scores function

    # test case for checking truly random scores created
    def test_random_value(self):
        # verify truly random for each student id (even though reproducible via seed)
        random_scores_1 = evaluation.random_scores(
            1, self.processed_datasets["interaction_data"]
        )
        random_scores_2 = evaluation.random_scores(
            2, self.processed_datasets["interaction_data"]
        )

        # verify output is truly random, note that comparison on np arrays due to errors with ambigious (as .values converts to np array)
        self.assertFalse(
            np.array_equal(
                random_scores_1["random_score"].values,
                random_scores_2["random_score"].values,
            )
        )

    # test case for checking range of random scores is 0-1
    def test_random_range(self):
        # perform function to be tested
        random_scores = evaluation.random_scores(
            1, self.processed_datasets["interaction_data"]
        )

        # score column values between 0 and 1 using appropriate assert methods
        self.assertGreaterEqual(1, random_scores["random_score"].values)
        self.assertLessEqual(0, random_scores["random_score"].values)

    # 4. f1 score helper function

    # test case for checking 0 result of f1 evaluation metric if precision/recall is 0
    def test_f1_zero(self):
        # set to 0
        precision = 0
        recall = 0

        # perform function to be tested
        f1_score = evaluation.f1_score_helper(precision, recall)

        # should return 0 if both precision and recall 0 (no division by 0 error)
        self.assertEqual(f1_score, 0)

    # test case for checking general result of f1 evaluation metric
    def test_f1_value(self):
        # arbitrary precision/recall values
        precision = 0.3
        recall = 0.5

        # perform function to be tested
        f1_score = evaluation.f1_score_helper(precision, recall)

        # formula is 2 * ((precision*recall)/(precision+recall)) = 2 * ((0.3*0.5)/(0.3+0.5)) = 0.375 via calculator
        # using almost equal because or test failure where the value is 0.3749999
        self.assertAlmostEqual(f1_score, 0.375)

    # 5. ml temporal split function

    # test case for checking training vle data is cutoff correctly
    def test_ml_temporal_split_training_cutoff(self):

        # setting cutoff days 1 and 3 (both exist as per sample data)
        training_history_vle, training_future_vle, test_history_vle, test_future_vle = (
            evaluation.ml_temporal_split(
                self.processed_datasets["vle_data"], training_cutoff=1, test_cutoff=3
            )
        )

        # training history = date <= trainin gcutoff
        self.assertTrue((training_history_vle["date"] <= 1).all())

        # training future = test cutoff >= date > training cutoff
        self.assertTrue(
            (
                (training_future_vle["date"] <= 3) & (training_future_vle["date"] > 1)
            ).all()
        )

    # test case for checking test vle data is cutoff correctly
    def test_ml_temporal_split_test_cutoff(self):

        # setting cutoff days 1 and 3 (both exist as per sample data)
        training_history_vle, training_future_vle, test_history_vle, test_future_vle = (
            evaluation.ml_temporal_split(
                self.processed_datasets["vle_data"], training_cutoff=1, test_cutoff=3
            )
        )

        # test history = date <= test cutoff
        self.assertTrue((test_history_vle["date"] <= 3).all())

        # test future = date > test cutoff
        self.assertTrue((test_future_vle["date"] > 3).all())

    # 6. aggregate interaction data function

    # test case for correct columns of aggregated interactions
    def test_aggregate_interaction_columns(self):

        # perform function to be tested
        aggregate_result = evaluation.aggregate_interaction_data(
            vle_data=self.processed_datasets["vle_data"]
        )

        # all new variables created via aggregation + existing student/resource info (note that the order remains same as creation to prevent test fail)
        self.assertEqual(
            [
                "id_student",
                "code_module",
                "code_presentation",
                "id_site",
                "activity_type",
                "total_resource_clicks",
                "first_used",
                "last_used",
                "click_duration",
            ],
            aggregate_result.columns.to_list(),
        )

    # test case for correct values for aggregated interactions based on sample data
    def test_aggregate_interaction_values(self):

        # perform function to be tested
        aggregate_result = evaluation.aggregate_interaction_data(
            vle_data=self.processed_datasets["vle_data"]
        )

        # student 2 has one intereaction with resource site 201, clicks=14 at day=5
        student_2_site_201 = aggregate_result[
            (aggregate_result["id_student"] == 2) & (aggregate_result["id_site"] == 201)
        ].iloc[0]

        # check these values are correct as expected
        self.assertEqual(student_2_site_201["total_resource_clicks"], 14)
        self.assertEqual(student_2_site_201["first_used"], 5)
        self.assertEqual(student_2_site_201["last_used"], 5)
        # click duration is last used - first used = 5 -5 = 0
        self.assertEqual(student_2_site_201["click_duration"], 0)

    # not testing evaluate baselines/classifier/recommenders because they simple use library established/already tested functions
