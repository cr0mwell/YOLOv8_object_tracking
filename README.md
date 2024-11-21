# Object Tracking (YOLOv8)
Current application performs multiple object tracking for the detected people apart from optional segmentation/pose estimation tasks using [YOLOv8](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md) models.<br>
It takes the _*.mp4_ video as an input, preprocesses it depending on the selected model parameters, runs the inference and outputs
the resulting video in _*.avi_ format.<br> It also stores the model output in _*.json_ format.<br>
Simple Online Realtime Tracking ([SORT](https://arxiv.org/abs/1602.00763)) algorithm was chosen to perform objects tracking.

### Multi object tracking example
<img src="media/track-example.gif" width="900" height="700" /><br>

### Pose estimation example
<img src="media/pose-example.gif" width="900" height="700" /><br>

### Instance segmentation example
<img src="media/seg-example.gif" width="900" height="700" />

NOTE: videos for the above examples were taken at [pexels.com](https://www.pexels.com/search/videos/athlete)

### CLI options
Through CLI interface user can select the model type which in its turn implies the specific task:
* Detection only
* Detection and segmentation
* Detection and pose estimation

It's possible to choose the model size (see the details below) as well as the output visualization options:
* Show the processed video frame-by-frame during the inference process
* Show the output video after the input is completely processed

# Environment
Current application has been built on:
* Ubuntu 22.04
* C++ standard v17
* gcc >= 11.4 (installed by default)
* cmake >= 3.22 (see the [installation instructions](https://cmake.org/resources/))
* make >= 4.3 (installed by default)
* OpenCV >= 4.9 (see the [installation instructions](https://docs.opencv.org/4.x/d2/de6/tutorial_py_setup_in_ubuntu.html))
* Onnxruntime 1.20 ([included](https://github.com/cr0mwell/YOLOv8_object_tracking/blob/main/include/onnxruntime-linux-x64-1.20.0) in this repository)
* [Json](https://github.com/nlohmann/json) ([included](https://github.com/cr0mwell/YOLOv8_object_tracking/blob/main/include) in this repository)
* Google test >= 1.15 (deployed as a submodule)

## Installation
1. Clone this repo
    ```shell
   cd ~
   git clone https://github.com/cr0mwell/YOLOv8_object_tracking.git --recurse-submodules
   ```
2. Follow the instructions in the [script](https://colab.research.google.com/drive/10VorRTy0GfQqADt6ht0YNArsLUrSDJc4)
to convert the original YOLOv8 model by [Ultralytics](https://www.ultralytics.com/) and save it into `~/YOLOv8_object_tracking/models/yolo` folder.<br>
Keep the following model name convention: `YOlOv8<model_size>[-<seg>|<pose>].onnx` (F.e. _YOLOv8n-seg.onnx_, _YOLOv8l-pose.onnx_)
3. As an optional step, place your own video to process to the `~/YOLOv8_object_tracking/media` directory. Rename it to `input.mp4`. 
4. Make a build directory in the top level directory: `cd ~/YOLOv8_object_tracking && mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run and enter the runtime options from command prompt: `./object_tracking`
5. Processed video can be found in `~/YOLOv8_object_tracking/media/output.avi`
6. `~/YOLOv8_object_tracking/json/session_dump.json` file contains all the model output results.
7. Optionaly run the unit tests: `./test` although the test coverage leaves much to be desired ;-)

# License
This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/cr0mwell/YOLOv8_object_tracking/blob/main/LICENSE) file for details.