# FP Project - Hybrid Interpretable Educational Content Recommender System, with Support for Common Recommender Issues

## Constraints:
- Due to GitHub's file size limits, the raw datasets must be downloaded and uploaded to the data/raw folder before execution.
- The provided deployment runs the FastAPI application locally using Uvicorn.
- The following steps use Conda for environment management, so ensure Conda is installed.
- All commands must be run from the root directory of the project.
- Development was conducted using Visual Studio Code on a MacBook M4 Pro, with Python version 3.12.

## Project Structure
- "data" folder stores raw and processed datasets, alongside specific data for the API, ML, and cross validation evaluation.
- "images" folder stores data visualizations generated during the data exploration stage.
- "models" folder stores serialized model files for the Random Forest and SVC ML models.
- "notebooks" folder stores Jupyter Notebooks used during the coding phase for iterative development.
- "outputs" folder stores individual and summary evaluation results for the baselines and ML models.
- "src" folder stores all the Python code to run the project, with the subfolders of "data_handling" containing initial data science tasks, "recommenders" containing the 5 main recommender models, and "output" containing the deployment display.
- "tests" folder stores all unit tests for data handling, recommenders, API, and pipeline tasks.

## Instructions for Running the Project:

### 1. Configure Datasets
- Visit the Kaggle URL: https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad/data
- Download the OULAD dataset.
- Place all 7 of its CSV files into the "data/raw" folder.
- The final "data/raw" folder should contain: "assessments.csv", "courses.csv", "studentAssessment.csv", "studentInfo.csv", "studentRegistration.csv", "studentVle.csv" and "vle.csv" .

### 2. Create Environment with Dependencies
- Open a terminal and navigate to the root folder.
- Create a Conda environment with the dependencies by running: conda env create -f environment.yml -n <environment_name> 
- Activate the environment by running: conda activate <environment_name>
- Keep terminal open for the next steps.

### 3. Run Key Functions
- In the same terminal within the root folder, run: python -m src.output.run 
- This executes key data science tasks (may take a few minutes to finish execution, please wait until it is done loading the datasets/models required for recommendation generation in the following steps).

### 4. Launch API
- In the same terminal within the root folder, after the Python file execution is complete, run: uvicorn src.output.api:app  
- This starts FastAPI.

### 5. Navigating API
- In any browser, to view instructions, visit the URL: http://127.0.0.1:8000 
- To view routes, visit the URL: http://127.0.0.1:8000/docs

### 6. Sending Request
- In the same routes URL browser, under the "/recommendations" GET route, click the "Try it out" button. 
- Input the student ID, code module, and code presentation fields.
- Click "Execute" to generate and view the top 20 recommendations.

### 7. Running Unit Tests
- Open a terminal and navigate to the root folder.
- Activate the created Conda environment by running: conda activate <environment_name>
- Execute tests by running: python -m unittest discover -s tests -p 'test_*.py'

### 8. Additional Configuration (Optional)
- To run specific functions related to model evaluation, training, and generating recommendations, visit the "run.py" file within the "src/output" folder.
- Change the Boolean values for the required tasks to "True".
- For execution, in a terminal within the root folder, run: python -m src.output.run 