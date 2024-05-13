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

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

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

    # Seteamos los valores para mascara blanca
    lower_blanco = np.array([0, 0, 145])
    upper_blanco = np.array([160, 42, 209])

    # Seteamos los valores para mascara amarilla
    lower_amarillo = np.array([85, 50, 158])
    upper_amarillo = np.array([95, 238, 255])

    hsv = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    mask_blanco = cv2.inRange(hsv, lower_blanco, upper_blanco)
    mask_amarillo = cv2.inRange(hsv, lower_amarillo, upper_amarillo)

    # Definimos kernel
    kernel = np.ones((5,5),np.uint8)

    # Aplicamos operaciones morfologicas a blanco
    op_morf_blanco = cv2.erode(mask_blanco,kernel,iterations = 1)
    op_morf_blanco = cv2.dilate(op_morf_blanco,kernel,iterations = 2)

    # Aplicamos operaciones morfologicas a amarillo
    op_morf_amarillo = cv2.erode(mask_amarillo,kernel,iterations = 2)
    op_morf_amarillo = cv2.dilate(op_morf_amarillo,kernel,iterations = 2)

    # Filtro Canny en lineas blancas
    bordes_canny_blanco = cv2.Canny(op_morf_blanco, 180, 540)

    bordes_canny_blanco = cv2.cvtColor(bordes_canny_blanco, cv2.COLOR_GRAY2BGR)
    # Definir el kernel para la dilatación
    kernel_dilatacion = np.ones((3, 3), np.uint8)

    # Aplicar la dilatacion a las lineas blancas
    bordes_canny_blanco_dilatadas = cv2.dilate(bordes_canny_blanco, kernel_dilatacion, iterations=1)

    # Filtro Canny en lineas amarillas
    bordes_canny_amarillo = cv2.Canny(op_morf_amarillo, 180, 540)

    # Transformada de Hough probabilistica
    linesP = cv2.HoughLinesP(bordes_canny_amarillo, 1, np.pi / 180, 5, None, 50, 40)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(obs, (l[0], l[1]), (l[2], l[3]), (0,255,255), 3, cv2.LINE_AA)
        
    # Unir los resultados en obs
    obs = cv2.bitwise_or(obs,bordes_canny_blanco_dilatadas)

    # Mostramos las imagenes generadas
    cv2.imshow('Detector de lineas',op_morf_amarillo)
    cv2.imshow('Detector de ',bordes_canny_blanco)
    cv2.imshow('Detector',bordes_canny_amarillo)
    cv2.waitKey(1)


    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
