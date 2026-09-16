# FP Project - Hybrid Interpretable Educational Content Recommender System, with Support for Common Recommender Issues

## Constraints:
- Due to GitHub's file size limits, the raw datasets must be downloaded and uploaded to the data/raw folder before execution.
- The provided deployment runs the FastAPI application locally using Uvicorn.

## Project Structure
- " data " folder stores raw and processed datasets, alongside specific data for the API, ML, and cross validation evaluation.
- " images " folder stores data visualizations generated during the data exploration stage.
- " models " folder stores serialized model files for the Random Forest and SVC ML models.
- " notebooks " folder stores Jupyter Notebooks used during the coding phase for iterative development.
- " outputs " folder stores individual and summary evaluation results for the baselines and ML models.
- " src " folder stores all the Python code to run the project, with the subfolders of " data_handling " containing initial data science tasks, " recommenders " containing the 5 main recommender models, and " output " containing the deployment display.
- " tests " folder stores all unit tests for data handling, recommenders, API, and pipeline tasks.

## Instructions for Running the Project:

### 1.Configure Datasets
- Visit the Kaggle " https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad/data " URL.
- Download the OULAD dataset.
- Place its 7 CSV files into the "data/raw" folder.

### 2. Create Environment with Dependencies
- Open a terminal and navigate to the root folder.
- Create a Conda environment with the dependenices by running " conda env create -f environment.yml -n <environment_name> ".
- Activate the environment by running " conda activate <environment_name> ".
- Keep terminal running.

### 3. Run Key Functions
- In the same terminal, run " python -m src.output.run " to execute key data science tasks (may take a few minutes to finish execution, please wait until it is done loading the datasets/models required for recommendation generation).

### 4. Launch API
- In the same terminal, after the Python file execution is complete, run " uvicorn src.output.api:app " to start FastAPI.

### 5. Navigating API
- In any browser, visit " http://127.0.0.1:8000 " URL.
- Go to " http://127.0.0.1:8000/docs " to view routes.

### 6. Sending Request
- In the same browser, under the " /recommendations " GET route, click the "Try it out" button. 
- Input the student ID, code module, and code presentation fields.
- Click execute to generate and view top 20 recommendations.

### 7. Running Unit Tests
- Open a terminal and navigate to the root folder.
- Activate the created virtual environment by running "conda activate <environment_name>".
- Run " python -m unittest discover -s tests -p 'test_*.py' " to run tests.

### 8. Additional Configuration (OPTIONAL)
- To run specific functions related to model evaluation, training, and generating recommendations, visit the " run " Python file within the " src/output " folder.
- Change the Boolean values for the required tasks to " True ".
- Run " python -m src.output.run " for execution.