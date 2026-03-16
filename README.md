# AAE4011 Assignment 1 — Q3: ROS-Based Vehicle Detection from Rosbag

> **Student Name:** GUO Feiyang | **Student ID:** 23103837D | **Date:** 2026/Mar/17

---

## 1. Overview

This project implements a ROS-based vehicle detection pipeline that reads camera images from a rosbag, runs object detection (YOLO), and displays bounding boxes with class labels and confidence scores. It includes scripts to extract all frames from a bag (with image count and properties reported), a subscription-based detector for use with `roslaunch` and `rosbag play`, and a standalone GUI that lets you pick a bag file, confirm video details in a popup, then run detection with progress bar, pause, seek, speed, and font-size controls.

---

## 2. Detection Method *(Q3.1 — 2 marks)*

**Model:** YOLOv8 (Ultralytics), pretrained on COCO, using the nano variant (`yolov8n.pt`) by default.

**Why this architecture:** YOLOv8 is a single-stage detector that gives a good trade-off between speed and accuracy, suitable for real-time or offline playback. Only COCO classes that correspond to vehicles (car, truck, bus, motorbike, motorcycle, train) are drawn; confidence threshold is adjustable via the UI. The choice supports the grading criteria: bounding boxes, class labels, and confidence scores are clearly overlaid on each frame, and the same pipeline can be extended to run on a live camera or drone feed.

---

## 3. Repository Structure

```
aae4011_vehicle_detection/
├── CMakeLists.txt
├── package.xml
├── launch/
│   └── detect_from_bag.launch
├── scripts/
│   ├── extract_images.py
│   ├── vehicle_detector_node.py
│   └── vehicle_detector_gui_node.py
└── README.md
```

- **CMakeLists.txt** — Build and install configuration; installs Python scripts and launch files.
- **package.xml** — Package metadata and ROS dependencies (rospy, sensor_msgs, cv_bridge, image_transport, rosbag).
- **detect_from_bag.launch** — Starts `rosbag play` and the vehicle detector node; arguments: `bag_path`, `image_topic`.
- **extract_images.py** — Reads a rosbag, extracts all frames from the image topic, saves them as images, and reports image count and properties (resolution, topic).
- **vehicle_detector_node.py** — Subscribes to an image topic, runs YOLO detection, and displays results in an OpenCV window; for use with `roslaunch` and `rosbag play`.
- **vehicle_detector_gui_node.py** — Standalone GUI: file picker for .bag, popup with video details (image count, resolution, duration, topic); after user confirmation, runs detection with progress bar, current/total time, pause, seek, speed, and font-size controls.

---

## 4. Prerequisites

- **OS:** Ubuntu 20.04 (e.g. WSL2 on Windows 10/11) or native Linux.
- **ROS:** Noetic.
- **Python:** 3.8+.
- **ROS dependencies:** `rospy`, `sensor_msgs`, `cv_bridge`, `image_transport`, `rosbag` (from package.xml).
- **Python packages:**  
  `pip3 install ultralytics opencv-python`  
  (For image extraction and GUI, `rosbag` is used via the ROS environment.)

---

## 5. How to Run *(Q3.1 — 2 marks)*

1. **Clone the repository** (or copy the package into your catkin workspace):
   ```bash
   cd ~/catkin_ws/src
   git clone <your-repo-url>
   # or copy aae4011_vehicle_detection/ into src/
   ```

2. **Install dependencies:**
   ```bash
   pip3 install ultralytics opencv-python
   ```

3. **Build the ROS package:**
   ```bash
   cd ~/catkin_ws
   catkin_make
   source devel/setup.bash
   ```

4. **Place the rosbag file** (e.g. on Desktop or in a known path). In WSL, Windows paths are under `/mnt/c/...` (e.g. `/mnt/c/Users/User/Desktop/2026-02-02-17-57-27.bag`).

5. **Launch the pipeline** — choose one of the following.

   **Option A — Launch file (rosbag + detector):**  
   Start `roscore` in one terminal, then in another:
   ```bash
   source ~/catkin_ws/devel/setup.bash
   roslaunch aae4011_vehicle_detection detect_from_bag.launch bag_path:=/path/to/your.bag image_topic:=/hikcamera/image_2/compressed
   ```

   **Option B — GUI node (file picker, popup details, then detection):**  
   In one terminal (with or without `roscore`):
   ```bash
   source ~/catkin_ws/devel/setup.bash
   rosrun aae4011_vehicle_detection vehicle_detector_gui_node.py
   ```
   Then select the .bag file in the dialog; confirm the video details in the popup; detection starts paused — press **Space** to play. Use the on-screen controls (progress bar, trackbars for confidence, speed, font size) as needed.

   **Image extraction only (reports image count & properties):**
   ```bash
   rosrun aae4011_vehicle_detection extract_images.py --bag /path/to/your.bag --topic /hikcamera/image_2/compressed --out_dir extracted_frames
   ```
   The script prints image count, resolution, and topic to the terminal.

---

## 6. Sample Results

- **Image extraction summary:**  
  *[After running `extract_images.py`, paste the terminal output here, e.g.:]*  
  - Total frames: 1142  
  - Resolution: 1920 x 1080 px  
  - Topic: `/hikcamera/image_2/compressed`

- **Detection results:**  
  *[Add a screenshot of the detection UI (bounding boxes, labels, confidence, statistics) and optionally a short sentence on detection statistics.]*

---

## 7. Video Demonstration *(Q3.2 — 5 marks)*

**Video Link:** [YouTube (Unlisted)](https://youtu.be/JiUkYiSbrw4)

The video (1–3 min) shows:
- (a) Launching the ROS package (e.g. running the GUI node and selecting a bag).
- (b) The UI displaying detection results (progress bar, time, bounding boxes, labels, confidence, statistics).
- (c) A brief explanation of the results (what is detected, how the controls work).

---

## 8. Reflection & Critical Analysis *(Q3.3 — 8 marks, 300–500 words)*

### (a) What Did You Learn? *(2 marks)*

I gained two main technical skills. First, I learned how to work with ROS bags and image topics in practice: reading a rosbag in Python, locating the `sensor_msgs/CompressedImage` topic, and either extracting every frame to disk with correct reporting of image count and properties or feeding frames into a detection node. Using `cv_bridge` and decoding compressed images into OpenCV format for YOLO became a clear pipeline. Second, I learned how to integrate a deep learning model (YOLOv8) into a ROS workflow—loading the model once, running inference per frame, and drawing bounding boxes, class labels, and confidence scores on the image for a functional UI that meets the assignment’s visualisation requirements.

### (b) How Did You Use AI Tools? *(2 marks)*

I used AI assistants to speed up package layout, launch file syntax, and parts of the detection and GUI code. The benefits were faster setup of `package.xml` and `CMakeLists.txt`, correct use of `$(arg ...)` in launch files, and a clear structure for the GUI node (file picker, popup, progress bar, trackbars). A clear limitation was that generated code had to be adapted to my environment (e.g. WSL paths, ROS Noetic, and the exact topic names in the provided bag). I still had to read and debug the scripts to understand how they work and to fix issues such as missing launch files or substitution errors. So AI tools helped with structure and boilerplate, but understanding and verifying the behaviour remained my responsibility.

### (c) How to Improve Accuracy? *(2 marks)*

Two concrete strategies would be: (1) **Use a larger or fine-tuned model.** Switching from YOLOv8n to YOLOv8m or YOLOv8l would improve recall and precision at the cost of speed. Better still, fine-tuning YOLOv8 on a dataset of aerial or roadside vehicle images would better match the viewpoint and scale of the target application (e.g. drone or dashcam), reducing missed and false detections. (2) **Improve input quality and tune the confidence threshold.** Preprocessing (e.g. denoising, contrast adjustment, or super-resolution for small vehicles) can make inputs easier for the model. Then tuning the confidence threshold (e.g. via the GUI trackbar) on a validation set would yield a better trade-off between precision and recall for the specific use case.

### (d) Real-World Challenges *(2 marks)*

Two main challenges for deploying this pipeline on an actual drone in real time are: (1) **Latency and onboard compute.** Inference must complete within the sensor’s frame interval (e.g. 10–30 Hz) so that detections are timely for control or logging. On a constrained drone platform, this may require a smaller model (e.g. YOLOv8n), quantization, or hardware acceleration (GPU/TPU). Encoding and transmission of images or results add further delay and must be considered in the system design. (2) **Robustness under real-world conditions.** Lighting changes, motion blur, small or partially occluded vehicles, and varying backgrounds can degrade accuracy compared to offline rosbag playback. Deploying on a real drone would require testing in those conditions, possibly combining vision with other sensors (e.g. lidar or radar) and defining fallback behaviours when confidence is low or the scene is ambiguous.

---

## 9. References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ROS Noetic](https://wiki.ros.org/noetic)
- [cv_bridge](https://wiki.ros.org/cv_bridge)
- [Example README for Q3](https://github.com/weisongwen/AAE4011-S22526/blob/main/assignment_1/example_readme_Q3.md)
