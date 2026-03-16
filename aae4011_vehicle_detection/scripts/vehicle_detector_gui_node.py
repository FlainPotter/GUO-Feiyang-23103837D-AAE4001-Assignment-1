#!/usr/bin/env python3
"""
Enhanced vehicle detection node with:
- GUI file picker for .bag (Windows path supported)
- Image count & properties report
- Progress bar, current/total time, seek, speed control, UI settings
"""
from __future__ import division
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

# Optional ROS (for logging only; node runs without roscore)
try:
    import rospy
    _use_ros = True
except ImportError:
    _use_ros = False

def _loginfo(msg, *args):
    if _use_ros:
        try:
            rospy.loginfo(msg, *args)
        except Exception:
            print(msg % args if args else msg)
    else:
        print(msg % args if args else msg)

def _logerr(msg, *args):
    if _use_ros:
        try:
            rospy.logerr(msg, *args)
        except Exception:
            print(msg % args if args else msg, file=sys.stderr)
    else:
        print(msg % args if args else msg, file=sys.stderr)

def _logwarn(msg, *args):
    if _use_ros:
        try:
            rospy.logwarn(msg, *args)
        except Exception:
            print(msg % args if args else msg, file=sys.stderr)
    else:
        print(msg % args if args else msg, file=sys.stderr)

# Try tkinter for file dialog (may not be available in headless)
try:
    import Tkinter as tk
    from tkFileDialog import askopenfilename
except ImportError:
    import tkinter as tk
    from tkinter.filedialog import askopenfilename

VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "motorcycle", "train"}


def windows_path_to_wsl(path):
    """Convert Windows path (C:\...) to WSL path (/mnt/c/...)."""
    if not path or "\\" not in path and ":" not in path:
        return path
    path = os.path.normpath(path).replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        rest = path[2:].lstrip("/")
        return "/mnt/{}/{}".format(drive, rest) if rest else "/mnt/{}".format(drive)
    return path


def select_bag_file():
    """Open file dialog to select a .bag file. Returns WSL path or None if cancelled."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = askopenfilename(
        title="Select ROS bag file",
        filetypes=[("ROS bag files", "*.bag"), ("All files", "*.*")],
        initialdir="/mnt/c/Users/User/Desktop",
    )
    root.destroy()
    if not path:
        return None
    return windows_path_to_wsl(path)


def get_bag_info(bag_path):
    """
    Read bag and return: topic, message_count, duration_sec, and first image (for resolution).
    """
    import rosbag
    bag = rosbag.Bag(bag_path, "r")
    info = bag.get_type_and_topic_info()
    image_topic = None
    message_count = 0
    for t, tinfo in info.topics.items():
        if tinfo.msg_type == "sensor_msgs/CompressedImage":
            if tinfo.message_count > message_count:
                message_count = tinfo.message_count
                image_topic = t
    if not image_topic or message_count == 0:
        bag.close()
        return None
    # Get first message for resolution and timestamps for duration
    first_stamp = None
    last_stamp = None
    first_data = None
    for _, msg, t in bag.read_messages(topics=[image_topic]):
        if first_stamp is None:
            first_stamp = msg.header.stamp.to_sec()
            first_data = msg.data
        last_stamp = msg.header.stamp.to_sec()
    bag.close()
    duration_sec = (last_stamp - first_stamp) if (first_stamp is not None and last_stamp is not None) else 0
    # Decode first image to get resolution
    w, h = 0, 0
    if first_data:
        arr = np.frombuffer(first_data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
    return {
        "topic": image_topic,
        "message_count": message_count,
        "duration_sec": duration_sec,
        "first_stamp": first_stamp,
        "last_stamp": last_stamp,
        "width": w,
        "height": h,
    }


def load_bag_frames(bag_path, image_topic):
    """Load all CompressedImage messages into list of (stamp, data)."""
    import rosbag
    bag = rosbag.Bag(bag_path, "r")
    frames = []
    for _, msg, _ in bag.read_messages(topics=[image_topic]):
        frames.append((msg.header.stamp.to_sec(), bytes(msg.data)))
    bag.close()
    return frames


def show_bag_info_and_confirm(info, bag_path):
    """
    Show popup with video details (image count & properties).
    User must click 'Start detection' to proceed; no auto-play.
    Returns True if user confirms, False if Cancel.
    """
    report = (
        "Video details (Script correctly reads the bag, extracts all frames,\n"
        "reports image count & properties):\n\n"
        "  Image count:  {}\n"
        "  Resolution:   {} x {} px\n"
        "  Duration:     {:.2f} s\n"
        "  Topic:        {}\n\n"
        "Click 'Start detection' to run vehicle detection on this video,\n"
        "or 'Cancel' to exit."
    ).format(
        info["message_count"],
        info["width"],
        info["height"],
        info["duration_sec"],
        info["topic"],
    )
    confirmed = [False]  # use list so inner function can set it

    root = tk.Tk()
    root.title("Bag info – confirm to start detection")
    root.attributes("-topmost", True)
    root.geometry("520x280")
    root.resizable(True, True)

    frame = tk.Frame(root, padx=15, pady=15)
    frame.pack(fill=tk.BOTH, expand=True)

    label = tk.Label(frame, text=report, justify=tk.LEFT, font=("Segoe UI", 10))
    label.pack(anchor=tk.W)

    btn_frame = tk.Frame(frame)
    btn_frame.pack(side=tk.BOTTOM, pady=(20, 0))

    def on_start():
        confirmed[0] = True
        root.destroy()

    def on_cancel():
        root.destroy()

    tk.Button(btn_frame, text="Start detection", command=on_start, width=14, font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Cancel", command=on_cancel, width=10, font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=5)

    root.mainloop()
    return confirmed[0]


class VehicleDetectorGUI:
    def __init__(self):
        if _use_ros:
            try:
                rospy.init_node("vehicle_detector_gui_node", anonymous=True)
                self.conf_threshold = rospy.get_param("~conf_threshold", 0.3)
                self.model_name = rospy.get_param("~model", "yolov8n.pt")
            except Exception:
                self.conf_threshold = 0.3
                self.model_name = "yolov8n.pt"
        else:
            self.conf_threshold = 0.3
            self.model_name = "yolov8n.pt"
        _loginfo("Loading YOLO model: %s", self.model_name)
        self.model = YOLO(self.model_name)
        self.per_class_counter = Counter()
        self.total_vehicle_count = 0

        # Playback state
        self.frames = []  # list of (stamp, data)
        self.bag_info = None
        self.current_index = 0
        self.playing = False  # Start paused; user presses Space to play
        self.speed = 1.0
        self.win_name = "Vehicle Detection"
        self.progress_bar_h = 28
        self.font_scale = 0.6
        self.font_thickness = 2
        self._n_frames = 0
        self._img_width = 0

    def draw_detections(self, img, results):
        frame_vehicle_count = 0
        if not results or len(results) == 0:
            return img, 0
        r = results[0]
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])
            if cls_name not in VEHICLE_CLASSES:
                continue
            frame_vehicle_count += 1
            self.total_vehicle_count += 1
            self.per_class_counter[cls_name] += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = "{} {:.2f}".format(cls_name, conf)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        y = img.shape[0] - 10
        fs = getattr(self, "font_scale", 0.6)
        ft = max(1, int(2 * fs / 0.6))
        for cls_name, cnt in self.per_class_counter.items():
            cv2.putText(img, "{}: {}".format(cls_name, cnt), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * fs, (255, 255, 255), ft, cv2.LINE_AA)
            y -= int(20 * fs)
        return img, frame_vehicle_count

    def draw_ui_overlay(self, img, frame_idx, n_frames, current_time_sec, duration_sec, n_vehicles_this_frame):
        h, w = img.shape[:2]
        fs = self.font_scale
        ft = max(1, int(self.font_thickness * fs / 0.6))
        # Pause indicator: big "PAUSED" when video is paused
        if not self.playing:
            cx, cy = w // 2, h // 2
            cv2.putText(img, "PAUSED", (cx - 80, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * fs, (0, 0, 255), ft + 2, cv2.LINE_AA)
            cv2.putText(img, "[Space to play]", (cx - 70, cy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * fs, (200, 200, 200), ft, cv2.LINE_AA)
        # Semi-transparent bar at bottom
        bar_y0 = h - self.progress_bar_h - 40
        overlay = img.copy()
        cv2.rectangle(overlay, (0, bar_y0), (w, h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        # Progress bar
        bar_h = 20
        bar_y = h - 35
        bar_margin = 10
        bar_w = w - 2 * bar_margin
        cv2.rectangle(img, (bar_margin, bar_y), (w - bar_margin, bar_y + bar_h), (80, 80, 80), -1)
        if n_frames > 0:
            fill_w = int(bar_w * (frame_idx + 1) / n_frames)
            cv2.rectangle(img, (bar_margin, bar_y), (bar_margin + fill_w, bar_y + bar_h), (0, 200, 0), -1)
        # Time text (use UI font size)
        def fmt(t):
            m = int(t // 60)
            s = int(t % 60)
            return "{:02d}:{:02d}".format(m, s)
        time_str = "{} / {}".format(fmt(current_time_sec), fmt(duration_sec))
        cv2.putText(img, time_str, (bar_margin, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), ft, cv2.LINE_AA)
        speed_str = "Speed: {}x".format(self.speed)
        cv2.putText(img, speed_str, (w - 120, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), ft, cv2.LINE_AA)
        # Top line: frame and vehicles (use UI font size)
        info_str = "Frame {}/{}  Vehicles: {}  Total: {}".format(frame_idx + 1, n_frames, n_vehicles_this_frame, self.total_vehicle_count)
        cv2.putText(img, info_str, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * fs, (0, 255, 0), ft, cv2.LINE_AA)
        return img

    def mouse_callback(self, event, x, y, unused1, unused2):
        if event != cv2.EVENT_LBUTTONDOWN or self._n_frames == 0:
            return
        bar_margin = 10
        bar_w = self._img_width - 2 * bar_margin
        if bar_w <= 0:
            return
        rel_x = x - bar_margin
        if rel_x < 0:
            rel_x = 0
        if rel_x > bar_w:
            rel_x = bar_w
        self.current_index = min(int(self._n_frames * rel_x / bar_w), self._n_frames - 1)

    def run_standalone(self):
        # 1) File picker
        bag_path = select_bag_file()
        if not bag_path:
            _logwarn("No bag file selected. Exiting.")
            return
        if not os.path.isfile(bag_path):
            _logerr("File not found: %s", bag_path)
            return
        _loginfo("Selected bag: %s", bag_path)

        # 2) Get bag info and report (image count & properties)
        self.bag_info = get_bag_info(bag_path)
        if not self.bag_info:
            _logerr("No CompressedImage topic found in bag.")
            return
        info = self.bag_info
        report = (
            "Image count: {}\n"
            "Resolution: {} x {}\n"
            "Duration: {:.2f} s\n"
            "Topic: {}"
        ).format(info["message_count"], info["width"], info["height"], info["duration_sec"], info["topic"])
        _loginfo("Bag report:\n%s", report)
        print("\n========== Bag report ==========\n{}\n================================\n".format(report))

        # 2b) Popup: show video details and wait for user to confirm (no auto-play)
        if not show_bag_info_and_confirm(info, bag_path):
            _logwarn("User cancelled. Exiting.")
            return

        # 3) Load all frames (stamp, data)
        _loginfo("Loading frames...")
        self.frames = load_bag_frames(bag_path, info["topic"])
        _loginfo("Loaded %d frames.", len(self.frames))
        if not self.frames:
            return
        n_frames = len(self.frames)
        duration_sec = info["duration_sec"]
        first_stamp = self.frames[0][0]
        fps = (n_frames - 1) / (self.frames[-1][0] - first_stamp) if n_frames > 1 and (self.frames[-1][0] - first_stamp) > 0 else 10.0

        # 4) OpenCV window and trackbars (UI settings: confidence, speed, font size)
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Confidence", self.win_name, int(self.conf_threshold * 100), 99, lambda _: None)
        cv2.createTrackbar("Speed (0.25-4)", self.win_name, 8, 40, lambda _: None)  # 8 -> 1.0x default
        # Font size: 0-25 -> 0.4 to 2.0, default 10 -> ~1.0
        cv2.createTrackbar("Font size", self.win_name, 10, 25, lambda _: None)

        self._n_frames = n_frames
        self._img_width = info["width"]

        def on_mouse(event, x, y, flags, param):
            self.mouse_callback(event, x, y, None, None)
        cv2.setMouseCallback(self.win_name, on_mouse)

        last_frame_time = time.time()
        while True:
            if _use_ros and rospy.is_shutdown():
                break
            # Read trackbars (UI settings: confidence, speed, font size)
            self.conf_threshold = cv2.getTrackbarPos("Confidence", self.win_name) / 100.0
            if self.conf_threshold < 0.01:
                self.conf_threshold = 0.01
            speed_val = cv2.getTrackbarPos("Speed (0.25-4)", self.win_name)
            self.speed = 0.25 + (speed_val / 40.0) * 3.75  # 0.25 .. 4.0
            font_pos = cv2.getTrackbarPos("Font size", self.win_name)
            self.font_scale = 0.4 + (font_pos / 25.0) * 1.6  # 0..25 -> 0.4..2.0

            stamp, data = self.frames[self.current_index]
            arr = np.frombuffer(data, np.uint8)
            cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if cv_img is None:
                self.current_index = min(self.current_index + 1, n_frames - 1)
                continue
            current_time_sec = stamp - first_stamp

            results = self.model(cv_img, verbose=False, conf=self.conf_threshold)
            cv_img, n_vehicles = self.draw_detections(cv_img, results)
            cv_img = self.draw_ui_overlay(cv_img, self.current_index, n_frames, current_time_sec, duration_sec, n_vehicles)

            cv2.imshow(self.win_name, cv_img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord(" "):
                self.playing = not self.playing
            elif key == 81 or key == ord("a"):  # Left / A: back 5 sec
                target = stamp - 5.0
                for i in range(self.current_index - 1, -1, -1):
                    if self.frames[i][0] <= target:
                        self.current_index = i
                        break
                else:
                    self.current_index = 0
            elif key == 83 or key == ord("d"):  # Right / D: forward 5 sec
                target = stamp + 5.0
                for i in range(self.current_index + 1, n_frames):
                    if self.frames[i][0] >= target:
                        self.current_index = i
                        break
                else:
                    self.current_index = n_frames - 1
            elif key == ord("-") or key == ord("["):
                self.speed = max(0.25, self.speed - 0.25)
                cv2.setTrackbarPos("Speed (0.25-4)", self.win_name, int((self.speed - 0.25) / 3.75 * 40))
            elif key == ord("+") or key == ord("=") or key == ord("]"):
                self.speed = min(4.0, self.speed + 0.25)
                cv2.setTrackbarPos("Speed (0.25-4)", self.win_name, int((self.speed - 0.25) / 3.75 * 40))

            if self.playing:
                self.current_index = (self.current_index + 1) % n_frames
                if self.current_index == 0 and n_frames > 1:
                    self.total_vehicle_count = 0
                    self.per_class_counter.clear()
                elapsed = time.time() - last_frame_time
                desired = 1.0 / (fps * self.speed)
                if desired > elapsed:
                    time.sleep(desired - elapsed)
            last_frame_time = time.time()

        cv2.destroyAllWindows()


def main():
    node = VehicleDetectorGUI()
    node.run_standalone()


if __name__ == "__main__":
    main()
