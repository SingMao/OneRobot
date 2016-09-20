# coding: utf-8

import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel

try:
    aa
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    from pylibfreenect2 import CpuPacketPipeline
    pipeline = CpuPacketPipeline()

# Create and set logger
# logger = createConsoleLogger(LoggerLevel.Warning)
# setGlobalLogger(logger)

class Kinect:
    def __init__(self):
        self.fn = Freenect2()
        num_devices = self.fn.enumerateDevices()
        if num_devices == 0:
            print("No device connected!")
            sys.exit(1)

        serial = self.fn.getDeviceSerialNumber(0)
        self.device = self.fn.openDevice(serial, pipeline=pipeline)

        self.listener = SyncMultiFrameListener(
            FrameType.Color)

        # Register listeners
        self.device.setColorFrameListener(self.listener)
        # device.setIrAndDepthFrameListener(listener)

        self.device.start()

        # NOTE: must be called after device.start()
        registration = Registration(self.device.getIrCameraParams(),
                                    self.device.getColorCameraParams())

    def __del__(self):
        self.device.stop()
        self.device.close()

    def get_image(self):
        frames = self.listener.waitForNewFrame()

        color = frames["color"].asarray()[:,::-1,:3]
        self.listener.release(frames)

        # if np.sum(color[-1,:,:]) == 0:
            # return None

        return color

