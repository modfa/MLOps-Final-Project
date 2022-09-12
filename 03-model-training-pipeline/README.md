For this part, we will use the prefect version 2.0b5 and use the AWS EC2 instance as prefect server as well as s3 bucket for the flow schedules stortage.

Create a conda environment and install the necessary packages using the ```requirement.txt``` file

Set up the Prefect Server and S3 bucket on AWS and the runb the script ```register_model.py```   (Refer the week-3 videos)
Check this link for all the details - https://gist.github.com/Qfl3x/8dd69b8173f027b9468016c118f3b6a5

Now we have the flow/task schedule in the s3 bucket which any agent can run on the targeted envirornment (Container, Kuberenetes etc )
