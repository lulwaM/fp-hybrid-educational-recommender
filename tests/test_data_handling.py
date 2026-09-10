#python testing framework
import unittest

#for data storage
import pandas as pd

#to create sample test data
from tests.sample_data import create_datasets

#functions to be tested
from src.data_handling import preprocessing
from src.data_handling import feature_engineering

class TestPreprocessing(unittest.TestCase):

    #runs once before tests, used to store datasets to be evaluated
    def setUp(self):
        self.datasets = create_datasets()

    #1. merge datasets function

    #test expected keys
    def test_merge_keys(self):
        merged_result = preprocessing.merge_datasets(self.datasets)

        self.assertEqual(merged_result.keys(),
                         {"student_data","assessment_data","vle_data","resource_data"})

    #test dataframe has expected columns/data
    def test_merge_student_dataframe_details(self):
        merged_result = preprocessing.merge_datasets(self.datasets)

        self.assertIn("id_student",merged_result["student_data"].columns)
        self.assertIn("code_module",merged_result["student_data"].columns)
        self.assertIn("gender",merged_result["student_data"].columns)
        self.assertIn("disability",merged_result["student_data"].columns)

    #2. clean datasets function

    #handling missing values in student data for registration data by dropping records
    def test_clean_drop_missing_student_registration(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)

        #any method returns true if any of the registration dates are NaN, vectorized operation
        self.assertFalse(cleaned_result["student_data"]["date_registration"].isna().any())

    #handling missing values in student data for registration/unregistration date and imd band by dropping/filling records
    def test_clean_missing_student(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)

        #any method returns true if any of the registration dates are NaN, vectorized operation
        self.assertFalse(cleaned_result["student_data"]["date_registration"].isna().any())
        self.assertFalse(cleaned_result["student_data"]["imd_band"].isna().any())
        self.assertFalse(cleaned_result["student_data"]["date_unregistration"].isna().any())

    #handling missing values in assessment data for scores/date by dropping/filling records
    def test_clean_missing_assessment(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)

        #any method returns true if any of the registration dates are NaN, vectorized operation
        self.assertFalse(cleaned_result["assessment_data"]["score"].isna().any())
        self.assertFalse(cleaned_result["assessment_data"]["date"].isna().any())

    #handling missing values in vle data for dropping entire week from/to columns (does not exist)
    def test_clean_missing_vle(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)

        self.assertNotIn("week_from",cleaned_result["vle_data"])
        self.assertNotIn("week_to",cleaned_result["vle_data"])

    #3. typecasting datasets function

    #handling categorical columns data type conversion
    def test_typecast_categorical(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)

        #only selected some key categorical columns to test, note that assert instance used here because pandas variable is a class

        #for student data
        self.assertIsInstance(typecast_result["student_data"]["code_module"].dtype, pd.CategoricalDtype)
        self.assertIsInstance(typecast_result["student_data"]["gender"].dtype, pd.CategoricalDtype)
        self.assertIsInstance(typecast_result["student_data"]["final_result"].dtype, pd.CategoricalDtype)
        #for assessment data
        self.assertIsInstance(typecast_result["assessment_data"]["assessment_type"].dtype, pd.CategoricalDtype)
        #for vle data
        self.assertIsInstance(typecast_result["vle_data"]["activity_type"].dtype, pd.CategoricalDtype)
        #for resource data
        self.assertIsInstance(typecast_result["resource_data"]["code_presentation"].dtype, pd.CategoricalDtype)

    #handling boolean columns data type conversion
    def test_typecast_boolean(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)

        #only selected some key boolean columns to test, note that assert equal used here because python bool is object

        #for student data
        self.assertEqual(typecast_result["student_data"]["withdrew"].dtype, bool) 
        self.assertEqual(typecast_result["student_data"]["disability"].dtype, bool) 
        #for assessment data
        self.assertEqual(typecast_result["assessment_data"]["is_banked"].dtype, bool)

    #4. processing datasets function

    def test_processing_student(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)
        processed_result = preprocessing.processing_datasets(typecast_result)

        #code inspired by: https://stackoverflow.com/questions/40913113/compare-two-pandas-series-dataframes-that-are-virtually-equal
        #verify feature created is correct according to formula
        pd.testing.assert_series_equal(processed_result["student_data"]["registration_duration"],processed_result["student_data"]["date_unregistration"]-processed_result["student_data"]["date_registration"],check_names=False )
        #end inspired code

    def test_processing_assessment(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)
        processed_result = preprocessing.processing_datasets(typecast_result)

        self.assertIn("average_score",processed_result["assessment_data"].columns)

    def test_processing_resource(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)
        processed_result = preprocessing.processing_datasets(typecast_result)

        #code inspired by: https://stackoverflow.com/questions/40913113/compare-two-pandas-series-dataframes-that-are-virtually-equal
        #verify feature created is correct according to formula
        pd.testing.assert_series_equal(processed_result["resource_data"]["resource_text"],
                                       processed_result["resource_data"]["code_module"].astype(str) + " " + processed_result["resource_data"]["code_presentation"].astype(str) + " " + processed_result["resource_data"]["activity_type"].astype(str), check_names=False )
        #end inspired code

    def test_processing_interaction(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)
        processed_result = preprocessing.processing_datasets(typecast_result)

        self.assertIn("total_resource_clicks",processed_result["interaction_data"])

        #code inspired by: https://stackoverflow.com/questions/40913113/compare-two-pandas-series-dataframes-that-are-virtually-equal
        #verify feature created is correct according to formula
        pd.testing.assert_series_equal(processed_result["interaction_data"]["click_duration"],
                                       processed_result["interaction_data"]["last_used"]-processed_result["interaction_data"]["first_used"], check_names=False )
        #end inspired code


    # now feature engineering functions

    # 5. filter data by cutoff function

    def test_filter_data_cutoff(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)
        processed_result = preprocessing.processing_datasets(typecast_result)

        cutoff_result = feature_engineering.filter_data_by_cutoff(datasets=processed_result,cutoff_day=10)

        # assert features for dates are <= cutoff day of 10, using all method because ocnverts to boolean value true if condition holds for all records
        self.assertTrue((cutoff_result["assessment_data"]["date_submitted"]<=10).all())

        self.assertTrue((cutoff_result["vle_data"]["date"]<=10).all())

        self.assertTrue((cutoff_result["student_data"]["date_registration"]<=10).all())


    def test_filter_data_leakage(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)
        processed_result = preprocessing.processing_datasets(typecast_result)

        cutoff_result = feature_engineering.filter_data_by_cutoff(datasets=processed_result,cutoff_day=10)

        # interaction data removed to prevent leakage
        self.assertNotIn("interaction_data",cutoff_result)

        # specific long term features removed to prevent leakage that is unintended

        self.assertNotIn("registration_duration",cutoff_result["student_data"].columns)
        self.assertNotIn("final_result",cutoff_result["student_data"].columns)
        self.assertNotIn("withdrew",cutoff_result["student_data"].columns)
        self.assertNotIn("date_unregistration",cutoff_result["student_data"].columns)

        self.assertNotIn("average_score",cutoff_result["assessment_data"].columns)


    # 6. create assessment features function

    def test_assessment_features_created(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)
        processed_result = preprocessing.processing_datasets(typecast_result)

        assessment_features = feature_engineering.create_assessment_features(assessment_data=processed_result["assessment_data"])

        # check for the feature engineered columns
        self.assertIn("average_score",assessment_features.columns)
        self.assertIn("min_score",assessment_features.columns)
        self.assertIn("max_score",assessment_features.columns)
        self.assertIn("assessments_submitted",assessment_features.columns)
        self.assertIn("assessment_type_diversity",assessment_features.columns)

    def test_assessment_features_values(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)
        processed_result = preprocessing.processing_datasets(typecast_result)

        assessment_features = feature_engineering.create_assessment_features(assessment_data=processed_result["assessment_data"])

        # looking at sample data, these should be the values for student 2's engineered columns (only 1 assessment submitted with score of 60)
        # getting first row of features (only 1)
        student_2_features = assessment_features[assessment_features["id_student"]==2].iloc[0]

        # check expected values
        self.assertEqual(student_2_features["average_score"],60)
        self.assertEqual(student_2_features["min_score"],60)
        self.assertEqual(student_2_features["max_score"],60)
        self.assertEqual(student_2_features["assessments_submitted"],1)


    # 7. create vle features function

    def test_vle_features_created(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)
        processed_result = preprocessing.processing_datasets(typecast_result)

        vle_features = feature_engineering.create_vle_features(vle_data=processed_result["vle_data"])

        # check for the feature engineered columns
        self.assertIn("total_resource_clicks",vle_features.columns)
        self.assertIn("average_resource_clicks",vle_features.columns)
        self.assertIn("first_used",vle_features.columns)
        self.assertIn("last_used",vle_features.columns)
        self.assertIn("resources_used",vle_features.columns)
        self.assertIn("interaction_duration",vle_features.columns)


    def test_vle_features_values(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)
        processed_result = preprocessing.processing_datasets(typecast_result)

        vle_features = feature_engineering.create_vle_features(vle_data=processed_result["vle_data"])

        # looking at sample data, these should be the values for student 2's engineered columns (only 1 interaction 14 clicks used day=5 so duration is 5-5=0)
        # getting first row of features (only 1)
        student_2_features = vle_features[vle_features["id_student"]==2].iloc[0]

        # check expected values
        self.assertEqual(student_2_features["total_resource_clicks"],14)
        self.assertEqual(student_2_features["total_resource_clicks"],14)
        self.assertEqual(student_2_features["resources_used"],1)
        self.assertEqual(student_2_features["first_used"],5)
        self.assertEqual(student_2_features["last_used"],5)
        self.assertEqual(student_2_features["interaction_duration"],0)

    # 8. create features function

    def test_create_features_created(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)
        processed_result = preprocessing.processing_datasets(typecast_result)

        features = feature_engineering.create_features(datasets=processed_result,cutoff_day=10)

        # assesment features present
        self.assertIn("average_score",features.columns)
        self.assertIn("assessments_submitted",features.columns)

        # vle features present
        self.assertIn("total_resource_clicks",features.columns)
        self.assertIn("first_used",features.columns)

        # custom student feature present
        self.assertIn("days_registered",features.columns)

    def test_create_features_cutoff(self):

        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)
        processed_result = preprocessing.processing_datasets(typecast_result)

        features = feature_engineering.create_features(datasets=processed_result,cutoff_day=10)

        # date feature less than or equal cutoff
        self.assertTrue((features["date_registration"]<=10).all())

        # no leaking column
        self.assertNotIn("date_unregistration",features.columns)


    # not testing exploration file functions becuase only data visualizations and is based on already tested functions/data

#code copied from: https://docs.python.org/3/library/unittest.html
#used to run tests
if __name__ == '__main__':
    unittest.main()
#end copied code
    

