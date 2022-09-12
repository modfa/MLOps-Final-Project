## For The experiment tracking, we will use the AWS EC2 instance as MLFLOW server and also use s3 bucket for the artifacts store (all the codes in the script)


### 1) Install MLflow

To get started with MLflow you'll need to install the MLflow Python package.

For this we recommend creating a separate Python environment, for example, you can use conda environments, and then install the package/s (requirement.txt file) there with pip or conda.

Once you installed the package, run the command ```mlflow --version``` and check the output.



2 ) execute this command:

  python project_script.py  --raw_data_path <CARS_DATA_FOLDER> --dest_path .

3 ) The script ```training_mlflow.py``` will load the datasets produced by the previous step, train the model on the training set and finally calculate the RMSE on the validation set.

4 ) Launch the tracking server on AWS and we use the S3 bucket for artifacts storage.

5) Tune the hyperparameters of the model

Now let's try to reduce the validation error by tuning the hyperparameters of the random forest regressor using hyperopt.

We have prepared the script ```hpo.py``` for this exercise. 
run this script.


6 ) Promote the best model to the model registry as discussed in the week-2 lectures.
