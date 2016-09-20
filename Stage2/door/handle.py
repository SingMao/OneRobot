from __future__ import print_function
import numpy as np
import rospy
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
import time
import matplotlib.pyplot as plt
import cv2

DEPTH_TOPIC = '/kinect2/sd/image_depth_rect'
# IMAGE_TOPIC = '/kinect2/sd/image_color_rect'
# CLOUD_TOPIC = '/rtabmap/cloud_map'
# CLOUD_TOPIC = '/kinect2/sd/points'
# ODOM_TOPIC = '/rtabmap/odom'

def point_cloud_to_numpy(msg):
    dtype_list = [('x', np.float32), ('y', np.float32), ('z', np.float32),
                  ('__dummy1', np.float32),
                  ('r', np.uint8), ('g', np.uint8), ('b', np.uint8),
                  ('__dummy2', np.uint8), ('__dummy3', np.float32),
                  ('__dummy4', np.float32), ('__dummy5', np.float32)]

    arr = np.fromstring(msg.data, dtype_list)
    arr = arr[['x', 'y', 'z', 'r', 'g', 'b']]
    if msg.height == 1:
        return np.reshape(arr, (msg.width,))
    else:
        return np.reshape(arr, (msg.height, msg.width))

def cloud_cb(msg):
    # msg = numpy_msg(msg)
    msg = point_cloud_to_numpy(msg)
    arr = msg[~np.isnan(msg['x'])]
    points = np.vstack((arr['x'], arr['y'], arr['z'])).T
    get_handle(points)
    # print(points.shape)
    # print(rawpoints.shape)
    # print(type(msg))

def odom_cb(msg):
    print(msg.pose.pose.position)
    print(msg.pose.pose.orientation)
    print()

def image_cb(msg):
    arr = np.fromstring(msg.data, np.uint8).reshape((msg.width, msg.height, 3))
    print(arr.shape)

def get_handle(points):
    print(len(points))
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    idx = np.argsort(x)
    fir = idx[:10]
    print(fir)
    print(points[fir])
    print('x', np.mean(x), np.max(x), np.min(x))
    print('y', np.mean(y), np.max(y), np.min(y))
    print('z', np.mean(z), np.max(z), np.min(z))

    plt.plot(x, y)

def to_jet(img):
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = (np.maximum(0, np.minimum(1, img)) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

def depth_cb(msg):
    # print(msg.height, msg.width, msg.step)
    arr = np.fromstring(msg.data, np.uint16).reshape((msg.height, msg.width))

    nzcnt = np.sum(arr > 0)

    mean = np.sum(arr) / nzcnt

    img = (arr - mean + 100) / 300.

    img_int = (img*255).astype(np.uint8)

    smooth = cv2.medianBlur(img_int, 51)
    # diff = (img_int.astype(np.float32) - smooth.astype(np.float32)) / 255 * 3 + 0.5
    diff = img_int.astype(np.float32) - smooth.astype(np.float32)
    near = (diff < 10).astype(uint8)

    print(nzcnt, np.sum(arr) / nzcnt)

    cv2.imshow('depth', to_jet(img_int))
    cv2.imshow('smooth', to_jet(smooth))
    cv2.imshow('diff', near)
    cv2.waitKey(50)
    


def run():
    # cloudsub = rospy.Subscriber(CLOUD_TOPIC, PointCloud2, cloud_cb)
    # odomsub = rospy.Subscriber(ODOM_TOPIC, Odometry, odom_cb)
    # imagesub = rospy.Subscriber(IMAGE_TOPIC, Image, image_cb)
    imagesub = rospy.Subscriber(DEPTH_TOPIC, Image, depth_cb)
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('rostesting_node')
    print('starting...')
    try:
        run()
    except KeyboardInterrupt:
        print('break')
