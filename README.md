# Implementing Machine Learning at the Edge - A BEng Project
This is the repository for my project. Here the associated files with the project will be published, alongside any script descriptions that will be required to allow reproducability of the project. Only final scripts are included. Intermediatery scripts are not published.   
  
The project runs on TensorFlow 2.3.1 so please ensure this is the version used!    
  
Final demonstration video can be found here: https://www.youtube.com/watch?v=mZBnqr2PrLE  
Sendr web-application for model transfer GitHub: https://github.com/mazabdul7/Sendr    
  
TensorFlow Guide: https://www.tensorflow.org/guide

## Model and Dataset Files
All developed models .h5 files can be found here. Also included are the split datasets used for training in accordance with the 72/18/10 split used in the project.  
https://drive.google.com/drive/folders/1am20SxrEkWRJFyJPXvZrTgII-ZZoe2yj?usp=sharing    
The drive breakdown is as follows: 
### Model Files
- EfficientNetB0-Default.h5 - First developed WasteNet model.
- EfficientNetB0-Small.h5 - WasteNet model at 75% resolution.
- EfficientNetB0-PureIBM.h5 - WasteNet model trained purely on IBM dataset.
- EfficientNetB0-IBM.h5 - WasteNet model trained on mixed TrashNet & IBM dataset.
### Datasets (Pre-split)
- dataYoupengSplit.zip - TrashNet training dataset
- testtrashnet.zip - TrashNet test-set
- pureibm.zip - IBM training dataset
- testibm.zip - IBM test-set
- dataYoupentSplit_Mixed.zip - Mixed training dataset (TrashNet + IBM)

## Scripts
The final scripts developed for WasteNet can be found above. The training Jupyter notebook is also provided so that results can be reproduced. Each scripts description is given below.

### Training.ipynb
This is the training notebook run on Google Colab. This notebook is used to train, fine-tune and test the accuracy of models. It is commented and can be followed to reproduce the models.

### main.py
This is the main inference script run on the Jetson Nano. This script loads the TensorRT model and initialises the Raspberry PI camera module and servomotors. It then runs inference, displays the output on the screen and sends signals to the Servos accordingly.(Note the model may take up to 7 minutes to load)    
To run this ensure all required modules are installed on the Jetson Nano. The required modules are listed above.

### manual_acceleration.py
This is the TensorRT acceleration script. The model to be accelerated must be in SavedModel format. If needed, use Keras to load the .h5 model file and then save it as a SavedModel file. The script is already configured to accelerate the model for the fastest configuration. (Note: This script is for manual acceleration only, if pulling an update from Sendr run the update.py instead)

### update.py
This is the update script that would be run daily on the Jetson Nano. The script contacts the Sendr local server and downloads any models pushed. It then converts the model to a SavedModel format and accelerates it. Please configure accordingly with your local ip address of the host machine that is running the Sendr application. 

## Contact
If you need help with using any file please contact me here: zceebda@ucl.ac.uk
