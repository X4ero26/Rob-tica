#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
import cv2
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper



# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckietown_udem1')
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()


def nothing(x):
    pass

# Create a window
cv2.namedWindow('image')

# create trackbars for color change
#cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
#cv2.createTrackbar('HMax','image',0,179,nothing)

#cv2.createTrackbar('SMin','image',0,255,nothing)
#cv2.createTrackbar('SMax','image',0,255,nothing)

#cv2.createTrackbar('VMin','image',0,255,nothing)
#cv2.createTrackbar('VMax','image',0,255,nothing)

# Set default value for MAX HSV trackbars.
#cv2.setTrackbarPos('HMax', 'image', 179)
#cv2.setTrackbarPos('SMax', 'image', 255)
#cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

# img = cv2.imread("messi.jpg")
# img = cv2.resize(img,(640,480))
# output = img
# waitTime = 33

def red_alert(frame):
    red_img = np.zeros((480, 640, 3), dtype = np.uint8)
    red_img[:,:,2] = 90
    blend = cv2.addWeighted(frame, 0.5, red_img, 0.5, 0)

    return blend

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):\

    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

def update(dt):
    global hMin ,sMin ,vMin ,hMax ,sMax ,vMax 
    global phMin ,psMin ,pvMin ,phMax ,psMax ,pvMax 
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)

        im.save('screen.png')

    if done:
        print('done!')
        #env.reset()
        env.render()

    # get current positions of all trackbars
    #hMin = cv2.getTrackbarPos('HMin','image')
    #sMin = cv2.getTrackbarPos('SMin','image')
    #vMin = cv2.getTrackbarPos('VMin','image')

    #hMax = cv2.getTrackbarPos('HMax','image')
    #sMax = cv2.getTrackbarPos('SMax','image')
    #vMax = cv2.getTrackbarPos('VMax','image')

    # Set minimum and max HSV values to display
    #lower = np.array([hMin, sMin, vMin])
    #upper = np.array([hMax, sMax, vMax])

    # Seteamos los valores HSV
    lower = np.array([93, 254,  216])
    upper = np.array([96, 255, 255])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    output = cv2.bitwise_and(obs,obs)

    # Aplicamos operaciones morfologicas
    kernel = np.ones((5,5),np.uint8)
    op_morf = cv2.erode(mask,kernel,iterations = 2)
    op_morf = cv2.dilate(op_morf,kernel,iterations = 30)
    

    
    # Contornos

    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    # org 
    org = (50, 50) 
    
    # fontScale 
    fontScale = 1
    
    # Line thickness of 2 px 
    thickness = 2

    contours, _ = cv2.findContours(op_morf, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for countour in contours:
        x,y,w,h = cv2.boundingRect(countour)
        cv2.rectangle(output, (x,y), (x + w, y + h), (0,0,255), 2)            
        if w > 300 or h > 400:
            action = np.array([-0.44, 0])
            env.step(action)
            output = red_alert(output)
            cv2.putText(output, 'Freno de Emergencia', org, font,  
                   fontScale, (0,0,255), thickness, cv2.LINE_AA) 
        cv2.putText(output, 'Patito', (x,y), font,  
                fontScale, (0,0,255), thickness, cv2.LINE_AA) 
        
            

    # Print if there is a change in HSV value
    if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display output image
    cv2.imshow('image',output)
    cv2.waitKey(1)


    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()