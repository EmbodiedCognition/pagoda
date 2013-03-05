# Copyright (c) 2013 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''OpenGL world viewer.'''

import glumpy
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut
import sys

from . import base


class Null(object):
    def __init__(self, world):
        self.world = world

    def run(self):
        while self.world.step():
            self.world.trace()
            if self.world.needs_reset():
                self.world.reset()


class GL(glumpy.Figure):
    def __init__(self, world, trace=False, paused=False, distance=30):
        super(GL, self).__init__()
        self.world = world
        self.trace = trace
        self.paused = paused
        self.elapsed = 0
        self.lens = self.add_frame()
        self.trackball = glumpy.Trackball(65, 30, 1, distance)
        self._x = 0
        self._y = 0

    def noop(self, *args, **kwargs):
        pass

    on_mouse_press = noop
    on_mouse_release = noop
    on_mouse_motion = noop

    def on_mouse_scroll(self, x, y, dx, dy):
        paused = self.paused
        self.paused = True
        self.trackball.zoom_to(x, y, dx, 20 * [1, -1][dy < 0])
        self.redraw()
        self.paused = paused

    def on_mouse_drag(self, x, y, dx, dy, button):
        paused = self.paused
        self.paused = True
        if button == 1:  # pan_to
            self._x += 0.1 * dx
            self._y += 0.1 * dy
        else:
            self.trackball.drag_to(x, y, dx, dy)
        self.redraw()
        self.paused = paused

    def on_init(self):
        self.on_resize(800, 600)

        gl.glEnable(gl.GL_BLEND)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_NORMALIZE)

        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glLight(gl.GL_LIGHT0, gl.GL_POSITION, [2, 2, 5, 0.5])
        gl.glLight(gl.GL_LIGHT0, gl.GL_DIFFUSE, [1, 1, 1, 1])
        gl.glLight(gl.GL_LIGHT0, gl.GL_SPECULAR, [1, 1, 1, 1])
        gl.glEnable(gl.GL_LIGHT1)
        gl.glLight(gl.GL_LIGHT0, gl.GL_POSITION, [-2, 4, 5, 0.5])
        gl.glLight(gl.GL_LIGHT0, gl.GL_DIFFUSE, [1, 1, 1, 1])
        gl.glLight(gl.GL_LIGHT0, gl.GL_SPECULAR, [1, 1, 1, 1])

        gl.glShadeModel(gl.GL_FLAT)

    def on_resize(self, width, height):
        w, h = float(width), float(height)

        gl.glViewport(0, 0, int(w), int(h))

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(60, w / h, 2, 300)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glMultMatrixf(self.trackball.matrix)

    def on_key_press(self, key, modifiers):
        if key == glumpy.window.key.ESCAPE:
            sys.exit()
        if key == glumpy.window.key.SPACE:
            self.paused = False if self.paused else True
        else:
            self.world.reset()
        self.redraw()

    def on_draw(self):
        self.clear(0, 0, 0)

        # modified slightly from glumpy.trackball.Trackball.push
        _, _, w, h = gl.glGetIntegerv(gl.GL_VIEWPORT)
        top = np.tan(35 * np.pi / 360) * 0.1 * self.trackball.zoom
        right = w * top / float(h)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glFrustum(-right, right, -top, top, 0.1, 100)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glTranslate(self._x, self._y, -self.trackball.distance)
        gl.glMultMatrixf(self.trackball.matrix)

        # draw a square on the ground plane.
        gl.glBegin(gl.GL_QUADS)
        gl.glColor(0.2, 0.3, 0.4)
        gl.glNormal(0, 0, 1)
        gl.glVertex(-10,  10, 0)
        gl.glVertex( 10,  10, 0)
        gl.glVertex( 10, -10, 0)
        gl.glVertex(-10, -10, 0)
        gl.glEnd()

        self.world.draw()

        self.trackball.pop()

    def on_idle(self, dt):
        if self.paused:
            return
        self.elapsed += dt
        while self.elapsed > self.world.dt:
            self.elapsed -= self.world.dt
            if not self.world.step():
                sys.exit()
            if self.trace:
                self.world.trace()
            if self.world.needs_reset():
                self.world.reset()
            self.redraw()

    def run(self):
        self.show()
