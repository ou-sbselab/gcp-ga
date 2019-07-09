"""This example spawns (bouncing) balls randomly on a L-shape constructed of 
two segment shapes. Not interactive.
"""

__version__ = "$Id:$"
__docformat__ = "reStructuredText"

# Python imports
import random

# Library imports
import pygame
from pygame.key import *
from pygame.locals import *
from pygame.color import *

# pymunk imports
import pymunkoptions
pymunkoptions.options['debug'] = False
import pymunk
import pymunk.pygame_util
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Executes Cloud-GA')
parser.add_argument('--x0', type=int)
parser.add_argument('--x1', type=int)
parser.add_argument('--x2', type=int)
parser.add_argument('--x3', type=int)
parser.add_argument('--y0', type=int)
parser.add_argument('--y1', type=int)
parser.add_argument('--y2', type=int)
parser.add_argument('--y3', type=int)
parser.add_argument('--emitX', type=int)
parser.add_argument('--emitY', type=int)
parser.add_argument('--id', type=str)
parser.add_argument('--save_frame', action='store_true')
parser.add_argument('--seed', type=int, default=1)

os.environ['SDL_AUDIODRIVER']            = 'dsp' # fix ALSA error
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'   # Hide PyGame messages

class BouncyBalls(object):
    """
    This class implements a simple scene in which there is a static platform (made up of a couple of lines)
    that don't move. Balls appear occasionally and drop onto the platform. They bounce around.
    """
    def __init__(self, args):#line1, line2):
        # Space
        self._space = pymunk.Space()
        self._space.gravity = (0.0, -900.0)

        self._args = args

        line1 = (args.x0, args.y0, args.x1, args.y1)
        line2 = (args.x2, args.y2, args.x3, args.y3)

        self._emitX = args.emitX
        self._emitY = args.emitY

        self._line1 = line1
        self._line2 = line2

        #print(self._line1,self._line2)

        # Physics
        # Time step
        self._dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1

        # pygame
        pygame.init()
        self._screen = pygame.display.set_mode((600, 600))
        self._clock = pygame.time.Clock()

        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()

        # Balls that exist in the world
        self._balls = []

        # Execution control and time until the next ball spawns
        self._running = True
        self._ticks_to_next_ball = 10

    def run(self):
        """
        The main loop of the game.
        :return: None
        """
        # Main loop
        while self._running:
            # Progress time forward
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)

            self._process_events()
            self._update_balls()
            self._clear_screen()
            self._draw_objects()
            pygame.display.flip()
            # Delay fixed time between frames
            self._clock.tick(50)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

            if pygame.time.get_ticks() > 60000:#120000:
                #pygame.image.save(self._screen, '%s-lastframe.png' % self._args.id)
                self._running = False

                if args.save_frame:
                  fname = '%d_%d_%d_%d_%d_%d_%d_%d_%d_%d.jpg' % (args.x0, args.y0, args.x1, args.y1, args.x2, args.y2, args.x3, args.y3, args.emitX, args.emitY)
                  pygame.image.save(self._screen, fname)
        return str(len(self._balls))

    def _add_static_scenery(self):
        """
        Create the static bodies.
        :return: None
        """
        static_body = self._space.static_body
        static_lines = [pymunk.Segment(static_body, (int(self._line1[0]),int(self._line1[1])), \
                                                    (int(self._line1[2]),int(self._line1[3])), 0.0),
                        pymunk.Segment(static_body, (int(self._line2[0]),int(self._line2[1])), \
                                                    (int(self._line2[2]),int(self._line2[3])), 0.0)]
        #static_lines = [pymunk.Segment(static_body, (111.0, 280.0), (407.0, 246.0), 0.0),
        #                pymunk.Segment(static_body, (407.0, 246.0), (407.0, 343.0), 0.0)]
        for line in static_lines:
            line.elasticity = 0.95
            line.friction = 0.9
        self._space.add(static_lines)

    def _process_events(self):
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: None
        """
        for event in pygame.event.get():
            if event.type == QUIT:
                self._running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                self._running = False
            elif event.type == KEYDOWN and event.key == K_p:
                pygame.image.save(self._screen, "bouncing_balls.png")

    def _update_balls(self):
        """
        Create/remove balls as necessary. Call once per frame only.
        :return: None
        """
        self._ticks_to_next_ball -= 1
        if self._ticks_to_next_ball <= 0:
            self._create_ball()
            self._ticks_to_next_ball = 100
        # Remove balls that fall below 100 vertically
        balls_to_remove = [ball for ball in self._balls if ball.body.position.y < 100]
        for ball in balls_to_remove:
            self._space.remove(ball, ball.body)
            self._balls.remove(ball)

    def _create_ball(self):
        """
        Create a ball.
        :return:
        """
        mass = 10
        radius = 25
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        #x = #random.randint(115, 350)
        body.position = self._emitX, self._emitY#x, 400
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 0.95
        shape.friction = 0.9
        self._space.add(body, shape)
        self._balls.append(shape)

    def _clear_screen(self):
        """
        Clears the screen.
        :return: None
        """
        self._screen.fill(THECOLORS["white"])

    def _draw_objects(self):
        """
        Draw the objects.
        :return: None
        """
        self._space.debug_draw(self._draw_options)

if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    game = BouncyBalls(args)

    # Print to stdout to return value back
    print(game.run())
