# Face detection and recognition
Image &amp; Video Analysis course 2020/21 at WUST repository.

## Results reproducibility
Follow the steps bellow:
1. Download the images from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71peklHb0pGdDl6R28) and unpack them all together into the `face-identification/images` directory.
2. Run the `process_images.py` script (you may want to change the parameters first, lines 18-22) to run the detection, alignment and embedding processes. This will take a while.
3. Run the `notebooks/DatasetAnalysis.ipynb` notebook to see some basic analyses. 
4. Run the `notebooks/DetectionEvaluation.ipynb` notebook to evaluate the detection results (this is necessary to find and save the filenames with correctly detected faces).
5. Run the `notebooks/Classification.ipynb` notebook to fix and save new identitites and evaluate the classifiers (this step can also take some time).

Next, if you have followed the steps above and want to run the camera test to see if the system can recognize your face, then:
* Create a directory `face-identification/my_images` and copy your own photos there.
* Run the `process_my_images.py` script.
* Run the `camera_test.py` script (but before change the `NAME` value to yours, line 9). 
* Enjoy.
