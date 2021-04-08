
# Project overview

for this "Object detection in an urban environment" project, we have performed exploratory data analysis. then we splitted the dataset into three parts training, testing and validation. In our case, we kept 75 data for the training and 15 for testing and validation. 

After editing the config file, we monitored the training and testing performance by using tensorboard. Then we tuned some hyperparameters in the config file to differentiate the performance with the previous training and evaluation process.

I have added several screenshots of my performance charts and detecting objects from our data. I have also added a video file to this repository.

All the files associated with this project can be found in this github repository https://github.com/hamzazia0/Object-Detection-in-an-Urban-Environment .


# Dataset

We used portion of the data from the Waymo Open dataset for this project. we downloaded the data using 'download_process.py' file. Kindly checkout the readme file to see the more detailed instruction.

## Dataset Analysis

In order to complete this step, we worked on the "Exploratory Data Analysis" notebook. We implemented the "display_instances" function to run the file.

After successfully executing the file, we can checkout some images where objects were succesfully detected by the program.
    

<p>
    <img src="images/eda_1.png"/>
    <br>
    <em>EDA_1</em>
</p>

<p>
    <img src="images/eda_2.png"/>
    <br>
    <em>EDA_2</em>
</p>

<p>
    <img src="images/eda_3.png"/>
    <br>
    <em>EDA_3</em>
</p>


## Cross Validation

