from __future__ import print_function
import numpy as np
import rospy
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image

import time
import threading

IMAGE_TOPIC = '/kinect2/sd/image_color_rect'

img_lock = threading.Lock()
img_data = None
ros_thread = None

def image_cb(msg):
    global img_data
    img_lock.acquire()
    img_data = np.fromstring(msg.data, np.uint8).reshape((msg.width, msg.height, 3))
    img_lock.release()
    print('ros E')

def run():
    print('zzzzzzzzzz')
    # rospy.init_node('rostesting_node', disable_signals=True)
    imagesub = rospy.Subscriber(IMAGE_TOPIC, Image, image_cb)
    rospy.spin()

def get_image():
    ret = None
    img_lock.acquire()
    ret = img_data
    img_lock.release()
    return ret

def start(cb):
    global ros_thread
    rospy.init_node('rostesting_node', disable_signals=True)
    ros_thread = threading.Thread(target=run)
    ros_thread.start()
    print('starting...')

    # import three
    try:
        while ros_thread.is_alive():
            cb()
    except KeyboardInterrupt:
        print('break')

    print('ROS Stop')
    stop()
    ros_thread.join()
    exit()
    time.sleep(1)

def stopped():
    global ros_thread
    return ros_thread is not None and not ros_thread.is_alive()

def stop():
    rospy.signal_shutdown('hao123')

if __name__ == '__main__':
    start()
