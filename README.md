# OCTclassification


This project applies transfer learning techniques to image classification tasks using TensorFlow and other associated libraries. Below is a brief overview of the core components and their roles in the project.
## Project Structure

### Main Attributes
- *dataset_main*: The primary dataset used for training and evaluating models.
- *imageSize*: Specifies the dimensions of the images utilized for training and inference.
- *model_name*: Indicates the name or type of the model used in transfer learning.
- *exp_config*: Experiment configuration settings.
- *conf*: Configuration parameters, for current training and evaluation run.

### Main Function
- **train**:
  - Responsible for training the model using transfer learning techniques. 
  - Encompasses steps such as data preprocessing, model initialization, training loop, and evaluation. 
  - Takes into account various parameters and configurations defined within the project.

## Setup and Usage

1. **Environment Setup**:
   - Python 3.9.
   - Install the required packages using:
     ```sh
     pip install -r requirements.txt
     ```

2. **Running the Project**:
   - To train the model, run the `transfer_learning.py` script:
     ```sh
     python transfer_learning.py
     ```

## Additional Information

This project leverages several powerful Python libraries including:
- **TensorFlow** for model training and deployment.
- **NumPy** for efficient numerical operations. 
- **Pandas** and **Matplotlib** for data manipulation and visualization respectively.

### Credentials

This repository holds the training setup used for research submitted to the journal *Electronics* for the article ** written by Tomasz Marciniak and Agnieszka Stankiewicz. 
Should you have any questions don't hessitate to contact us at: tomasz.marciniak [at] put.poznan.pl or agnieszka.stankiewicz [at] put.poznan.pl.