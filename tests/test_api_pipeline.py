# standard imports for python testing framework, mocking, testing API, data storage, and numerical calculations
import unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

# import to create sample test data and preprocess it
from tests.sample_data import create_datasets
from src.data_handling import preprocessing

# function to be tested
from src.output import pipeline

# prevents github action error by giving empty values for automatically executed function, code copied and adapted from: https://stackoverflow.com/questions/16134281/python-mocking-a-function-from-an-imported-module
with patch("src.data_handling.preprocessing.load_api_datasets", return_value={}), patch(
    "src.recommenders.ranking.load_model", return_value=None
):
    # function to be tested, must use same app
    from src.output.api import app
# end copied and adapted code


# test suite for API and pipeline files
# input: testcase from unittest library (required for testing) / no output, simply calls test cases for unit testing
class TestAPIPipeline(unittest.TestCase):

    # runs once before tests, used to store datasets to be evaluated (prevents code repetition)
    def setUp(self):

        # perform entire preprocessing pipeline on sample data
        self.datasets = create_datasets()
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)

        # final preprocessed dataset
        self.processed_datasets = preprocessing.processing_datasets(typecast_result)

        # code copied and adapted to work with unittest instead of pytest: https://fastapi.tiangolo.com/tutorial/testing/#using-testclient
        # setting up client for API
        self.client = TestClient(app)

    # 1. api routes functions

    # test case to check root message output and success
    def test_api_root(self):
        # access root page
        response = self.client.get("/")

        # check if no error returned, success
        self.assertEqual(response.status_code, 200)

        # check if main instruction present
        self.assertIn(
            "Please visit the /docs page to view routes and make a request.",
            response.json()["message"],
        )
        # end copied and adapted code

    # test case to check not successful request with invalid student id
    def test_api_invalid(self):
        # should be invalid because student id is not integer and not supplied additional path parameters
        response = self.client.get("/recommendations/invalid_student_id")

        # 200 is success, so should be anything other than that
        self.assertNotEqual(response.status_code, 200)

    # 2. pipeline generate recommendations function

    # test case to raise error with invalid student id
    def test_pipeline_invalid_student(self):

        # code copied from: https://stackoverflow.com/questions/129507/how-do-you-test-that-a-python-function-throws-an-exception
        with self.assertRaises(ValueError) as context:
            # id student 10 does not exist in sample data, so should error
            pipeline.generate_recommendations(
                id_student=10,
                code_module="AAA",
                code_presentation="2013J",
                datasets=self.processed_datasets,
                model=None,
                cutoff_day=86,
                k=20,
            )
        # end copied code

    # test case to raise error with invalid course due to no interactions
    def test_pipeline_invalid_course(self):

        # code copied from: https://stackoverflow.com/questions/129507/how-do-you-test-that-a-python-function-throws-an-exception
        with self.assertRaises(ValueError) as context:
            # id student 1 exists but has no interactions with course BBB / 2013B, so should error
            pipeline.generate_recommendations(
                id_student=1,
                code_module="BBB",
                code_presentation="2013B",
                datasets=self.processed_datasets,
                model=None,
                cutoff_day=86,
                k=20,
            )
        # end copied code

    # NOT testing other pipeline functions because they simply call/follow same process of already tested functions
