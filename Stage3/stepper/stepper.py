from __future__ import print_function
import serial
import struct
import argparse
import numpy as np
import time

ser = None

def rotate(yaw, pitch):
    print(yaw, pitch)
    ser.write('y')
    ser.write(struct.pack('f', yaw))
    ser.flush()
    ser.write('p')
    ser.write(struct.pack('f', pitch))
    ser.flush()

def init(port):
    global ser
    ser = serial.Serial(port, 115200, xonxoff=True)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

def run():
    while True:
        y, p = [float(x) * np.pi / 180 for x in raw_input().split()]
        rotate(y, p)

def turn_off():
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('port', help='serial port device')
    args = parser.parse_args()
    init(args.port)

    try:
        run()
    except KeyboardInterrupt:
        print('Catch Ctrl-C')

    turn_off()
