# Master Project

## :round_pushpin: General info
Hello, this is my master degree repository and contain main program. 

## Instuction for user
### System requierment
In order for the scripts and the necessary libraries to function correctly, the user must have a computer with the Linux operating system. In order to obtain performance similar to that presented in the work it is recommended to have a graphics card developed by NVIDIA.

### Programming language and libraries
Conde, a system for managing Python environments and packages, was used to facilitate the use of the scripts created. Conde should be installed according to the instructions on the manufacturer's website. The Python version and libraries were saved to the conda_environment.yml file in the "Install_Conda" folder. There is also a conda_env.py file that will download the necessary repository from the Github platform. Once the script has completed, the environment should be active.
```
$ python conda_env.py && conda activate stereovision
```

There is likely to be an error when running the actual script, so it is recommended that a change is made to the file "MobileSAM/mobile_sam/predictor.py"

``` Python
from mobile_sam.modeling import Sam --> from .modeling import Sam
```

### Taking images for calibration

Use the board saved as 'Plansza.png' to take the images for calibration. The taking_image.py script has been prepared to save images ready for calibration. It is very important to verify that the image displayed is from the correct camera, in other words the left image must come from the left camera. To save the images, place the board in front of both cameras, then the points are detected and the image can be saved. The image can be saved by pressing the s button, and the programme can be terminated with the q button. To obtain correct calibration results it is recommended to take at least 20 images.

### Running the main script

Once all the previous steps have been completed, run the main.py script. First check that the image displayed is from the correct camera. If the cameras have been correctly assigned, you can continue with the script. The result should be two windows displaying the image from both cameras, displaying a frame around the detected object together with the segmentation mask and the distance from the cameras. To terminate the programme, press the q button.
