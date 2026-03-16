#!/usr/bin/env python3
import os
import argparse

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge

def main():
    parser = argparse.ArgumentParser(
        description="Extract images from rosbag and save to disk."
    )
    parser.add_argument("--bag", required=True, help="Path to rosbag file")
    parser.add_argument(
        "--topic",
        default="/hikcamera/image_2/compressed",
        help="Image topic name (sensor_msgs/CompressedImage)",
    )
    parser.add_argument(
        "--out_dir",
        default="extracted_images",
        help="Output directory for saved images",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Opening bag: {args.bag}")
    bag = rosbag.Bag(args.bag, "r")

    bridge = CvBridge()
    count = 0
    width, height = None, None

    for topic, msg, t in bag.read_messages(topics=[args.topic]):
        # 解压 CompressedImage 到 OpenCV 图像
        cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

        if width is None:
            height, width = cv_img.shape[:2]

        filename = os.path.join(args.out_dir, f"frame_{count:06d}.png")
        cv2.imwrite(filename, cv_img)
        count += 1

        if count % 50 == 0:
            print(f"Saved {count} frames...")

    bag.close()

    print("====== Extraction Summary ======")
    print(f"Bag file    : {args.bag}")
    print(f"Topic       : {args.topic}")
    print(f"Output dir  : {os.path.abspath(args.out_dir)}")
    print(f"Image count : {count}")
    if width is not None:
        print(f"Resolution  : {width} x {height}")

if __name__ == "__main__":
    main()