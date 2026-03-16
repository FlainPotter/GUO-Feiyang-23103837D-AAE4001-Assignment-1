#!/usr/bin/env python3
import time
from collections import Counter

import cv2
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

from ultralytics import YOLO

VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "motorcycle", "train"}


class VehicleDetectorNode:
    def __init__(self):
        rospy.init_node("vehicle_detector_node")

        self.image_topic = rospy.get_param(
            "~image_topic", "/hikcamera/image_2/compressed"
        )
        self.conf_thres = rospy.get_param("~conf_threshold", 0.3)
        self.model_name = rospy.get_param("~model", "yolov8n.pt")

        rospy.loginfo(f"Loading YOLO model: {self.model_name}")
        self.model = YOLO(self.model_name)

        self.bridge = CvBridge()
        self.frame_count = 0
        self.total_vehicle_count = 0
        self.per_class_counter = Counter()
        self.last_time = time.time()

        self.sub = rospy.Subscriber(
            self.image_topic,
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo(f"Subscribed to image topic: {self.image_topic}")
        rospy.loginfo("Vehicle detector node initialised.")

    def image_callback(self, msg):
        self.frame_count += 1

        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        fps = 1.0 / dt if dt > 0 else 0.0

        cv_img = self.bridge.compressed_imgmsg_to_cv2(
            msg, desired_encoding="bgr8"
        )

        results = self.model(
            cv_img, verbose=False, conf=self.conf_thres
        )

        det_img, frame_vehicles = self.draw_detections(cv_img, results)

        text = f"Frame: {self.frame_count} | Vehicles in frame: {frame_vehicles} | Total: {self.total_vehicle_count} | FPS: {fps:.1f}"
        cv2.putText(
            det_img,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Vehicle Detection", det_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            rospy.signal_shutdown("User pressed ESC")

    def draw_detections(self, img, results):
        frame_vehicle_count = 0

        r = results[0]
        boxes = r.boxes
        names = r.names

        for box in boxes:
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
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        y = img.shape[0] - 10
        for cls_name, cnt in self.per_class_counter.items():
            text = f"{cls_name}: {cnt}"
            cv2.putText(
                img,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y -= 20

        return img, frame_vehicle_count

    def spin(self):
        rospy.loginfo("Vehicle detector is running. Press ESC in the window to quit.")
        rospy.spin()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    node = VehicleDetectorNode()
    node.spin()