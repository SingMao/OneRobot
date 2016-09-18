from __future__ import print_function
import numpy as np
import rospy
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2

CLOUD_TOPIC = '/rtabmap/cloud_map'
# CLOUD_TOPIC = '/kinect2/hd/points'
ODOM_TOPIC = '/rtabmap/odom'

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
    floor = arr[arr['z'] < 0]
    print(floor.shape)
    # print(rawpoints.shape)
    # print(type(msg))

def odom_cb(msg):
    print(msg.pose.pose.position)
    print(msg.pose.pose.orientation)
    print()

def run():
    cloudsub = rospy.Subscriber(CLOUD_TOPIC, PointCloud2, cloud_cb)
    # odomsub = rospy.Subscriber(ODOM_TOPIC, Odometry, odom_cb)
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('rostesting_node')
    print('starting...')
    try:
        run()
    except KeyboardInterrupt:
        print('break')
