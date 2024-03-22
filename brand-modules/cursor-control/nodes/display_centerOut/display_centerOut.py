#!/usr/bin/env python
import gc
import logging
import time
import os
import sys
from struct import unpack

import numpy as np
import pyglet
from brand import BRANDNode

# GRAPHICS
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
BLACK = (0, 0, 0)

# state definition
STATE_BETWEEN_TRIALS = 0
STATE_START_TRIAL = 1
STATE_MOVEMENT = 2
# Note: This section attempts to configure the display settings by reading the .DISPLAY file. 
# If graphical drivers are not installed or you encounter issues related to the display configuration,
# you can comment out this section to allow the experiment to run.
try:
    with open(os.path.join(os.path.expanduser('~'), '.DISPLAY'), 'r') as f:
        os.environ["DISPLAY"] = f.read().splitlines()[0]
except FileNotFoundError:
    logging.error('No display found, exiting')
    sys.exit(1)

class PygletDisplay(BRANDNode):

    def __init__(self):

        super().__init__()

        # initialize parameters
        self.fullscreen = self.parameters['fullscreen']
        self.window_width = self.parameters['window_width']
        self.window_height = self.parameters['window_height']
        self.syncbox_enable = self.parameters['syncbox']

        # window setup
        self.window = pyglet.window.Window(width=self.window_width,
                                           height=self.window_height,
                                           fullscreen=self.fullscreen)
        self.x_0 = int(self.window.width / 2)
        self.y_0 = int(self.window.height / 2)
        self.window.set_location(0, 0)
        # hide mouse
        self.window.set_mouse_visible(False)

        self.on_key_press = self.window.event(self.on_key_press)
        self.draw_stuff = self.window.event(self.draw_stuff)

        # create sprites
        self.batch = pyglet.graphics.Batch()
        self.background = pyglet.graphics.OrderedGroup(0)
        self.foreground = pyglet.graphics.OrderedGroup(1)

        # target
        self.target = pyglet.shapes.Circle(x=0,
                                           y=0,
                                           radius=40,
                                           color=RED,
                                           batch=self.batch,
                                           group=self.background)

        # sync box (graphical sync pulse for use with photodetector)
        self.syncbox = pyglet.shapes.Rectangle(x=0,
                                               y=0,
                                               width=200,
                                               height=200,
                                               color=WHITE,
                                               batch=self.batch,
                                               group=self.foreground)
        self.syncbox.y = self.window.height - self.syncbox.height

        # keypress label
        self.label = pyglet.text.Label(
            '',
            font_name=['Noto Sans', 'Times New Roman'],
            font_size=36,
            x=0,
            y=0,
            anchor_x='left',
            anchor_y='bottom',
            color=(125, 125, 125, 255),
            batch=self.batch,
            group=self.background)

        # cursor
        self.cursor = pyglet.shapes.Circle(x=0,
                                           y=0,
                                           radius=25,
                                           color=WHITE,
                                           batch=self.batch,
                                           group=self.foreground)

        # define timing and sync keys
        self.sync_key = self.parameters['sync_key'].encode()
        self.time_key = self.parameters['time_key'].encode()

        self.last_id = b'0-0'
        self.first_read = True

    # Getting data from Redis
    
    #In summary, this snippet of code is reading the most recent message from the 'mouse_ac' stream in Redis, accessing the data 
    #under the key 'samples', and then unpacking that binary data into three short integers. These likely represent information 
    #about the mouse, such as its position or movement.
    
    def get_mouse_position(self):
        
        # Reading data from Redis
        # 'mouse_ac' is the name of the Redis stream to read data from
        # '$' indicates to start reading from the latest message in the stream
        # count=1 means only the most recent message will be read
        # block=0 means the call will not block if there are no new messages
        reply = self.r.xread(streams={'mouse_ac': '$'}, count=1, block=0)
        
        # Accessing the response from Redis
        # reply[0] accesses the first tuple in the response list, corresponding to the 'mouse_ac' stream
        # reply[0][1] accesses the list of messages from that stream
        # reply[0][1][0] accesses the first message in that list
        # reply[0][1][0][1] accesses the content of the message, which is a dictionary of fields and values
        cursorFrame = reply[0][1][0][1]
        
        # Unpacking the message data
        # '3h' is the format for unpacking the data, indicating three short integers
        # cursorFrame[b'samples'] accesses the binary data associated with the 'samples' key in the message
        # The binary data is unpacked into a tuple of three short integers
        mouse_data = unpack('3h', cursorFrame[b'samples'])
        return mouse_data

    # Getting data from Redis
    def get_cursor(self):
        reply = self.r.xread(streams={'cursorData': '$'}, count=1, block=0)
        cursorFrame = reply[0][1][0][1]
        ups = {
            b'X': 'f',  # x position
            b'Y': 'f',  # y position
            b'radius': 'f',  # radius
            b'state': 'i'  # state
        }
        keys = [b'X', b'Y', b'radius', b'state']

        cursor_data = {}
        for key in keys:
            cursor_data[key.decode()] = unpack(ups[key], cursorFrame[key])[0]
        # print(f'cursor_data: {cursor_data}')  # for debugging
        return cursor_data

    def get_target(self):
        reply = self.r.xread(streams={'targetData': '$'}, count=1, block=0)
        targetFrame = reply[0][1][0][1]
        ups = {
            b'X': 'f',  # x position
            b'Y': 'f',  # y position
            b'radius': 'f',  # width
            b'state': 'i',  # state: hidden (0), yellow (1), green (2), red (3)
        }  # un-pack string
        keys = [b'X', b'Y', b'radius', b'state']

        target_data = {}
        for key in keys:
            target_data[key.decode()] = unpack(ups[key], targetFrame[key])[0]
        # print(f'target_data: {target_data}')  # for debugging
        return target_data

    # Pyglet event handlers
    
    # self.r: This is the Redis connection object. It's typically initialized elsewhere in the code (likely in the class's 
    #__init__ constructor).

    # xadd: This is a method to add a message to a stream in Redis.

    # b'keypress': This is the name of the stream in Redis. The 'b' prefix indicates that it's a byte string, which is necessary 
    #for keys and fields in Redis when using Python 3.

    # The following dictionary contains the fields and values to be added to the stream:
    # b'symbol': This is the byte string key for the field storing the symbol of the pressed key.
    # self.label.text: This is the value corresponding to the pressed key.
    # self.time_key: This is a byte string key referring to a timestamp field.
    # np.uint64(time.monotonic_ns()).tobytes(): This is the timestamp value converted to bytes.

    
    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:  # [ESC]
            self.window.close()
        else:
            self.r.xadd(
                b'keypress', {
                    b'symbol': self.label.text,
                    self.time_key: np.uint64(time.monotonic_ns()).tobytes()
                })

    def draw_stuff(self, *args):
        self.window.clear()

        self.tdict = self.get_target()
        self.cdict = self.get_cursor()

        # cursor position
        self.cursor.x, self.cursor.y = int(self.cdict['X'] +
                                           self.x_0), int(self.cdict['Y'] +
                                                          self.y_0)

        # target position
        self.target.x, self.target.y = int(self.tdict['X'] +
                                           self.x_0), int(self.tdict['Y'] +
                                                          self.y_0)

        # target color
        if self.tdict['state'] == 1:
            self.target.color = YELLOW
        elif self.tdict['state'] == 2:
            self.target.color = GREEN
        elif self.tdict['state'] == 3:
            self.target.color = RED

        # target visibility
        if self.tdict['state'] == 0:
            self.target.visible = False
            self.syncbox.visible = False and self.syncbox_enable
        else:
            self.target.visible = True
            self.syncbox.visible = True and self.syncbox_enable

        #self.center_mark.visible = self.center_mark_enable

        # cursor shape
        self.cursor.radius = int(self.cdict['radius'])

        # target shape
        self.target.radius = int(self.tdict['radius'])

        # log sync pulse state
        self.r.xadd(
            b'display_sync_pulse', {
                b'state': self.tdict['state'],
                self.time_key: np.uint64(time.monotonic_ns()).tobytes()
            })

        self.batch.draw()

    def terminate(self, sig, frame):
        pyglet.app.exit()
        super().terminate(sig, frame)

    def run(self):

        logging.info('Starting pyglet display...')

        pyglet.clock.schedule(self.draw_stuff)

        pyglet.app.run()


if __name__ == "__main__":
    gc.disable()

    # setup
    pyglet_display = PygletDisplay()

    # main
    pyglet_display.run()

    gc.collect()
