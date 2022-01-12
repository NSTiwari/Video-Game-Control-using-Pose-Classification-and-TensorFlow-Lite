# Video Game Control using Pose Classification and TensorFlow Lite

Following are the pre-requisites for this project to run.

- Ashes Cricket 2009 game
- Python 3.7 or higher version
- NumPy 1.20.0 or higher version
- OpenCV ```(opencv-python)``` 4.5.3.56 or higher version
- Pandas 1.3.1 or higher version
- TFLite Runtime ```(tflite-runtime)``` 2.7.0 or higher version

If you haven't already installed the above entities, kindly do so before moving ahead.

The ```setup.sh``` file will help you download the required libraries and dependencies.

Now, one common question that you will likely ask is:

Whether this project supports only the Ashes Cricket 2009 game? 
The answer is, no, you can integrate any game with this project. The controls of the keyboard action will change accordingly in the ```detect_pose.py``` file.

## Steps:

1. Clone the repository on your local machine.

2. Upload the ```Pose_Classification_using_TensorFlow_Lite.ipynb``` notebook on Google Colab.

3. If you wish to train the pose classifier for your custom dataset, the directory structure of your dataset should be of the following format:

```
cricket_poses/
    |__ train/
        |__ cover_drive/
            |______ cover_drive1.jpg
            |______ ...
    |__ test/
        |__ leg_glance/
            |______ leg_glance1.jpg
            |______ ...
```

3. Run the notebook cells one-by-one by following the instructions.

4. Download the ```pose_classifier.tflite``` model ```pose_labels.txt``` label file.

5. Copy the TF Lite model and label files in the repository you download in **Step 1** i.e., inside the ```Video-Game-Control-using-Pose-Classification-and-TensorFlow-Lite``` folder.

6. Install the **PyDirectInput** library using ```pip install PyDirectInput```.

7. Open the ```detect_pose.py``` file and edit **Line 40** by replacing ```<your-tflite-model>``` with the name of your TF Lite model (example: ```pose_classifier.tflite```). Agaon, edit **Line 41** by replacing ```<your-label-file>``` with the name of your label file (example: ```pose_labels.txt```).

8. Open command prompt and run ```detect_pose.py``` and enjoy playing the Ashes Cricket 2009 game by showing your cricket skills in front of the camera.

## Output:

![GitHub Logo](Output.gif)
