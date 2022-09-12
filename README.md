# MLOps-Final-Project

### Problem to solve--- Predicitng the Price of pre-owned Cars

There is one XYZ e-commerce company and they act as mediators between parties who are interested in selling or buying a pre owned cars. Now these are second hand cars and XYZ company act as mediators. Now in specific for the year 2015 to 2016, XYZ company have
recorded the data about the seller and car details.

So, what XYZ e-commerce company wants to do is they want to develop an algorithm, so that they can predict the price of the cars based on various attributes associated with the car.

So, XYZ e-commerce company has data for about 50,000 cars in their database that have been sold or that have been processed in one way another and there are these 19 variables that are associated with this problem.
Clearly, one of these variables is going to be the outcome variable or the price of the car and the other variables are variables that we hope have enough information in them so that we can
basically predict the price of the car.

The detailed description of all the variables can be found in the {} jupyter notebook where we explored the dataset in depth.


################################################


ENVIRONMENT SET-UP ON EC2 MACHINE/INSTANCE

Note --->  Recommended development environment: Linux 
For that we will use the AWS EC2 instance as our working station for exploring the dataset and training the model.

Step-1 )  Make sure you have the AWS account (you can create a free tier account ) and we will first  launch the EC2 instance (t2.micro – free tier or any other machine).

Step-2 ) Now we ssh into the EC2 instance using the CLI of our local system.

Step-3 )   Download and install the Anaconda distribution of Python

	wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
	bash Anaconda3-2022.05-Linux-x86_64.sh 

Step-4 ) Update existing packages
	
	sudo apt update

Step-5 )  Install Docker
  
  sudo apt install docker.io
  
  
Step-6 )  To run docker without sudo:
	sudo groupadd docker
	sudo usermod -aG docker $USER
	
	
Step-7) Install Docker Compose

Install docker-compose in a separate directory
	
	mkdir software
	cd software
Step-8 ) To get the latest release of Docker Compose, go to https://github.com/docker/compose and download the release for your OS.

	wget https://github.com/docker/compose/releases/download/v2.5.0/docker-compose-linux-x86_64 -O docker-compose

Step-9 ) Make it executable

	chmod +x docker-compose


Step-10 ) Add to the software directory to PATH. Open the .bashrc file with nano:

	nano ~/.bashrc

In .bashrc, add the following line:

	export PATH="${HOME}/software:${PATH}"

Save it and run the following to make sure the changes are applied:

	source .bashrc
	
Step-11 ) Run Docker

	docker run hello-world

If you get docker: Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/containers/create": dial unix /var/run/docker.sock: connect: permission denied. error, restart your VM instance.
Note: If you get It is required that your private key files are NOT accessible by others. This private key will be ignored. error, you should change permits on the downloaded file to protect your private key:

	chmod 400 name-of-your-private-key-file.pem

Step-12 ) Make sure everything is working as expected otherwise repeact the steps.

############################################################################

For Running the Notebooks/ Python Script ---

Note --> we will use the Python 3.9 version

1 ) Clone the repository from github into the EC2 instance
2) Now we will use the Visual Studio code editor for the workflow.
3) make sure to use the port forwarding so we can run the jupyter notebook on our local system.
 
Note – For any doubt related to setting up the environment,please watch the week-1 video (MLOps Zoomcamp 1.2 - Environment preparation )
 



