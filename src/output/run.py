# imports for task execution, API dataset preprocessing, and ranking ML model
from src.output import pipeline
from src.data_handling import preprocessing
from src.recommenders import ranking

# ------------------------------- FUNCTION CALLS FOR PIPELINE -----------------------------------------

# prevents running if imported
if __name__ == "__main__":

    # ------------------------------- REQUIRED FUNCTIONS -----------------------------------------

    # set boolean values of required tasks to run accordingly, DO NOT change because the rest of the functions depend on them
    bool_preprocess_datasets = True
    bool_create_API_datasets = True
    bool_create_visualizations = True
    bool_save_sample_recommendations = True

    if bool_preprocess_datasets:
        # 1. preprocessing for processed
        processed_datasets = pipeline.run_preprocessing()

    if bool_create_API_datasets:
        # 2. run to store api datasets, using cutoff as median of 86
        preprocessing.create_api_datasets(processed_datasets, 86)

    if bool_create_visualizations:
        # 3. generate visualizations
        pipeline.run_visualizations(processed_datasets)

    if bool_save_sample_recommendations:
        # 4. save sample recommendations
        model = ranking.load_model("models/random_forest_model.joblib")
        recommendations = pipeline.generate_recommendations(
            id_student=6516,
            code_module="AAA",
            code_presentation="2014J",
            datasets=processed_datasets,
            model=model,
            cutoff_day=86,
            k=20,
        )
        pipeline.save_sample_recommendations(recommendations, "sample_recommendations")

    # ------------------------------- OPTIONAL FUNCTIONS -----------------------------------------

    # set boolean values of optional tasks to run accordingly, edit parameters in conditional function calls

    # 1. run baseline evaluation for single or all students (hold out / cross validation)
    bool_baseline_single = False
    bool_baseline_holdout = False
    bool_baseline_cross_validation = False

    # 2. prepare ML data and train different ML model types (full data training only available for random forest because SVC will take days to train)
    # change parameters for sample size in function call below
    bool_prepare_ML_data = False
    bool_train_final_random_forest = False
    bool_train_sampled_random_forest = False
    bool_train_sampled_SVC = False

    # 3. run ML evaluation (different model types) as classifier/recommender + cross validation, sampled/full data (cross validation only available for random forest because too computaionally expensive for SVC)
    bool_evaluate_ML_recommender_random_forest = False
    bool_evaluate_ML_recommender_sampled_random_forest = False

    bool_evaluate_ML_classifier_random_forest = False
    bool_evaluate_ML_classifier_sampled_random_forest = False

    bool_evaluate_ML_recommender_sampled_SVC = False
    bool_evaluate_ML_classifier_sampled_SVC = False

    bool_evaluate_ML_cross_validation_random_forest = False

    # 4. generate recommendations with API data (no control over cutoff) or full processed data (to specify cutoff) using final random forest model
    # change parameters for evaluated student/code module/code presentation/cutoff day in function call below
    bool_generate_recommendations = False
    bool_generate_API_recommendations = False

    # conditionals that call functions according to boolean values, all using default values that may be changed:

    if bool_baseline_single:
        pipeline.run_baseline_evaluation_single(
            processed_datasets=processed_datasets,
            cutoff_day=86,
            evaluated_student=6516,
            k=20,
        )

    if bool_baseline_holdout:
        baseline_holdout_results, baseline_holdout_summary = (
            pipeline.run_baseline_evaluation_holdout(
                processed_datasets=processed_datasets, cutoff_day=150, k=20
            )
        )

    if bool_baseline_cross_validation:
        baseline_cv_results, baseline_cv_averages = (
            pipeline.run_baseline_cross_validation(
                processed_datasets=processed_datasets,
                cutoff_days=[60, 120, 180],
                test_size=60,
                k=20,
            )
        )

    if bool_prepare_ML_data:
        training_dataset, test_dataset = pipeline.run_prepare_ML_data(
            processed_datasets=processed_datasets,
            training_cutoff=86,
            test_cutoff=150,
            k=20,
        )

    if bool_train_final_random_forest:
        model = pipeline.run_train_ML(model_type="random_forest", sample_size=None)

    if bool_train_sampled_random_forest:
        random_forest_sample_model = pipeline.run_train_ML(
            model_type="random_forest", sample_size=10000
        )

    if bool_train_sampled_SVC:
        SVC_sample_model = pipeline.run_train_ML(model_type="SVC", sample_size=10000)

    if bool_evaluate_ML_recommender_random_forest:
        random_forest_results, random_forest_summary = (
            pipeline.run_evaluate_ML_recommender(
                processed_datasets=processed_datasets,
                file_name="random_forest_recommender_results",
                model_type="random_forest",
                model=None,
                model_path="models/random_forest_model.joblib",
                test_cutoff=150,
                k=20,
                saveData=True,
                saveSummary=True,
            )
        )

    # must initialize sample summary just in case saved in comparison table bool
    random_forest_sample_summary = None
    if bool_evaluate_ML_recommender_sampled_random_forest:
        random_forest_sample_results, random_forest_sample_summary = (
            pipeline.run_evaluate_ML_recommender(
                processed_datasets=processed_datasets,
                file_name="random_forest_sample_recommender_results",
                model_type="random_forest",
                model=None,
                model_path="models/random_forest_sample_model.joblib",
                test_cutoff=150,
                k=20,
                saveSummary=False,
            )
        )

    # must initialize sample summary just in case saved in comparison table bool
    SVC_sample_summary = None
    if bool_evaluate_ML_recommender_sampled_SVC:
        SVC_sample_results, SVC_sample_summary = pipeline.run_evaluate_ML_recommender(
            processed_datasets=processed_datasets,
            file_name="SVC_sample_recommender_results",
            model_type="SVC",
            model=None,
            model_path="models/SVC_sample_model.joblib",
            test_cutoff=150,
            k=20,
            saveSummary=False,
        )

    # can only save comparison if summaries for both models exist
    if random_forest_sample_summary is not None and SVC_sample_summary is not None:
        pipeline.run_save_ML_comparison_results(
            random_forest_sample_summary, SVC_sample_summary
        )

    if bool_evaluate_ML_classifier_random_forest:
        random_forest_classifier_results = pipeline.run_evaluate_ML_classifier(
            model=None,
            model_type="random_forest",
            model_path="models/random_forest_model.joblib",
            file_name="random_forest_classifier_results",
        )

    if bool_evaluate_ML_classifier_sampled_random_forest:
        random_forest_sample_classifier_results = pipeline.run_evaluate_ML_classifier(
            model=None,
            model_type="random_forest",
            model_path="models/random_forest_sample_model.joblib",
            file_name="random_forest_sample_classifier_results",
        )

    if bool_evaluate_ML_classifier_sampled_SVC:
        SVC_sample_classifier_results = pipeline.run_evaluate_ML_classifier(
            model=None,
            model_type="SVC",
            model_path="models/SVC_sample_model.joblib",
            file_name="SVC_sample_classifier_results",
        )

    if bool_evaluate_ML_cross_validation_random_forest:
        random_forest_cv_results, random_forest_cv_summary = (
            pipeline.run_evaluate_ML_cross_validation(
                processed_datasets=processed_datasets,
                cutoff_days=[60, 120, 180],
                test_size=60,
                k=20,
                model_type="random_forest",
            )
        )

    # to generate API recs, need both the model and API datasets
    if bool_generate_API_recommendations:
        model = ranking.load_model("models/random_forest_model.joblib")
        api_datasets = preprocessing.load_api_datasets()
        print(
            pipeline.generate_API_recommendations(
                id_student=6516,
                code_module="AAA",
                code_presentation="2014J",
                datasets=api_datasets,
                model=model,
            )
        )

    # to generate general recs, need only model as datasets created dynamically
    if bool_generate_recommendations:
        model = ranking.load_model("models/random_forest_model.joblib")
        print(
            pipeline.generate_recommendations(
                id_student=6516,
                code_module="AAA",
                code_presentation="2014J",
                datasets=processed_datasets,
                model=model,
                cutoff_day=86,
                k=20,
            )
        )
