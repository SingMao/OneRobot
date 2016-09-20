from __future__ import print_function
import serial
import struct
import argparse
import numpy as np
import time

V = 5
W = np.pi / 10.

ser = None

def stop():
    ser.write('s')
    ser.flush()

def set_speed(vec):
    ser.write('v')
    ser.write(struct.pack('fff', vec[0], vec[1], 0))
    ser.flush()

def set_omega(om):
    ser.write('v')
    ser.write(struct.pack('fff', 0, 0, om))
    ser.flush()

def move(x, y):
    if x == 0 and y == 0:
        return
    r = (x ** 2 + y ** 2) ** 0.5
    vec = np.array([x, y]) * V / float(r)
    t = r / float(V)
    set_speed(vec)
    time.sleep(t)
    stop()

def rotate(theta):
    om = W if theta >= 0. else -W
    t = theta / float(om)
    print('rotate waiting...', t)
    set_omega(om)
    time.sleep(t)
    stop()

def init(port):
    global ser
    ser = serial.Serial(port, 115200, xonxoff=True)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

def run():
    while True:
        ch = raw_input()
        if ch == 'v':
            x, y = [float(x) for x in raw_input().split()]
            move(x, y)
        elif ch == 'r':
            th = float(raw_input()) * np.pi / 180
            rotate(th)
        elif ch == 'q':
            ser.write(ch)
            ser.flush()
            break
        else:
            ser.write(ch)
        ser.flush()

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
