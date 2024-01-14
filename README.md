# Video Game Control using Pose Classification and TensorFlow Lite

Following are the pre-requisites for this project to run.

- Ashes Cricket 2009 game
- Python 3.7 or higher version
- NumPy 1.20.0 or higher version
- OpenCV ```(opencv-python)``` 4.5.3.56 or higher version
- Pandas 1.3.1 or higher version
- TFLite Runtime ```(tflite-runtime)``` 2.7.0 or higher version

If you haven't already installed the above entities, kindly do so before moving ahead.

The ```setup.sh``` file will help you download the required libraries and dependencies. Run it using ```sh setup.sh``` in Git Bash.

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

5. Copy the TF Lite model and label files in the repository you downloaded at **Step 1** i.e., inside the ```Video-Game-Control-using-Pose-Classification-and-TensorFlow-Lite``` folder.

6. Install the **PyDirectInput** library using ```pip install PyDirectInput```.

7. Open the ```detect_pose.py``` file and edit **Line 40** by replacing ```<your-tflite-model>``` with the name of your TF Lite model (example: ```pose_classifier.tflite```). Again, edit **Line 41** by replacing ```<your-label-file>``` with the name of your label file (example: ```pose_labels.txt```).

8. Open the Ashes Cricket 2009 game and change its display settings (```display.opt```) file to windowed mode and preferrably set the resolution to 1024x800.

8. Open command prompt and run ```detect_pose.py``` and enjoy playing the Ashes Cricket 2009 game by showing your batting skills in front of the camera.

## Demo Output:

![GitHub Logo](Output.gif)

- Watch the full demonstration on [YouTube](https://www.youtube.com/watch?v=Ubov6zZTuzI).

## References and Facts:
- This project is based on the [Human Pose Classification](www.tensorflow.org/lite/tutorials/pose_classification) tutorial by TensorFlow on a custom dataset of cricket shots.
- The [dataset](https://github.com/NSTiwari/Video-Game-Control-using-Pose-Classification-and-TensorFlow-Lite/blob/main/cricket_shots.zip) of the cricket shots - Cover Drive, Leg Glance and Cut were scraped from online sources and processed by various augmentations techniques.
- The Python file [```detect_pose.py```](https://github.com/NSTiwari/Video-Game-Control-using-Pose-Classification-and-TensorFlow-Lite/blob/main/detect_pose.py) developed by me, detects the pose and classifies the cricket shots and autonomously plays them in the video game.
- Papers and Blogs referred: [Cricket Shot Detection from Videos](https://ieeexplore.ieee.org/document/8494081) | [Detection Cricket Shots using Pose Estimation](https://blog.jovian.ai/detecting-cricket-shots-using-pose-estimation-8e69ed12fe98) | [Pose Estimation and Classification using TF Lite](https://blog.tensorflow.org/2021/08/pose-estimation-and-classification-on-edge-devices-with-MoveNet-and-TensorFlow-Lite.html).
