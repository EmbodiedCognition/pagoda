# Copyright (c) 2013 Leif Johnson <leif@cs.utexas.edu>
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

import climate
import contextlib
import copy
import numpy as np
import os
import pyglet

from pyglet.gl import *

logging = climate.get_logger(__name__)

from . import physics

TAU = 2 * np.pi


class Null(object):
    def __init__(self, world):
        self.world = world

    def run(self):
        while not self.world.step():
            if self.world.needs_reset():
                self.world.reset()


@contextlib.contextmanager
def gl_context(scale=None, translate=None, rotate=None, mat=None):
    glPushMatrix()
    if mat is not None:
        glMultMatrixf(vec(*mat))
    if translate is not None:
        glTranslatef(*translate)
    if rotate is not None:
        glRotatef(*rotate)
    if scale is not None:
        glScalef(*scale)
    yield
    glPopMatrix()


def vec(*args):
    return (GLfloat * len(args))(*args)


def build_vertex_list(idx, vtx, nrm):
    return pyglet.graphics.vertex_list_indexed(
        len(vtx) // 3, idx, ('v3f/static', vtx), ('n3f/static', nrm))


def box_vertices():
    vtx = np.array([
        [ 1, 1, 1], [ 1, 1, -1], [ 1, -1, -1], [ 1, -1, 1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, -1], [-1, -1, 1]], 'f')
    nrm = vtx / np.sqrt((vtx * vtx).sum(axis=1))[:, None]
    return [
        0, 3, 2,  0, 2, 1,  4, 5, 7,  7, 5, 6,  # x
        0, 1, 4,  4, 1, 5,  6, 2, 3,  6, 3, 7,  # y
        0, 4, 7,  0, 7, 3,  1, 2, 5,  5, 2, 6,  # z
    ], vtx.flatten(), nrm.flatten()


def sphere_vertices(n=3):
    idx = [[0, 1, 2], [0, 5, 1], [0, 2, 4], [0, 4, 5],
           [3, 2, 1], [3, 4, 2], [3, 5, 4], [3, 1, 5]]
    vtx = list(np.array([
        [ 1, 0, 0], [0,  1, 0], [0, 0,  1],
        [-1, 0, 0], [0, -1, 0], [0, 0, -1]], 'f'))
    for _ in range(n):
        idx_ = []
        for ui, vi, wi in idx:
            u, v, w = vtx[ui], vtx[vi], vtx[wi]
            d, e, f = u + v, v + w, w + u
            di = len(vtx)
            vtx.append(d / np.linalg.norm(d))
            ei = len(vtx)
            vtx.append(e / np.linalg.norm(e))
            fi = len(vtx)
            vtx.append(f / np.linalg.norm(f))
            idx_.append([ui, di, fi])
            idx_.append([vi, ei, di])
            idx_.append([wi, fi, ei])
            idx_.append([di, ei, fi])
        idx = idx_
    vtx = np.array(vtx, 'f').flatten()
    return np.array(idx).flatten(), vtx, vtx


def cylinder_vertices(n=14):
    idx = []
    vtx = [0, 0, 1,  0, 0, -1,  1, 0, 1,  1, 0, -1]
    nrm = [0, 0, 1,  0, 0, -1,  1, 0, 0,  1, 0, 0]
    thetas = np.linspace(0, TAU, n)
    for i in range(len(thetas) - 1):
        t0 = thetas[i]
        t1 = thetas[i+1]
        a = 2 * (i+1)
        b = 2 * (i+2)
        idx.extend([0, a, b,  a, a+1, b,  b, a+1, b+1,  b+1, a+1, 1])
        x, y = np.cos(t1), np.sin(t1)
        vtx.extend([x, y, 1,  x, y, -1])
        nrm.extend([x, y, 0,  x, y, 0])
    return idx, vtx, nrm


class EventLoop(pyglet.app.EventLoop):
    def run(self):
        self.has_exit = False
        self._legacy_setup()
        platform_event_loop = pyglet.app.platform_event_loop
        platform_event_loop.start()
        self.dispatch_event('on_enter')
        self.is_running = True
        while not self.has_exit:
            self.clock.tick()
            platform_event_loop.step(self.clock.get_sleep_time(True))
        self.is_running = False
        self.dispatch_event('on_exit')
        platform_event_loop.stop()

# use our event loop implementation rather than the default pyglet one.
pyglet.options['debug_gl'] = False
pyglet.app.event_loop = EventLoop()


class GL(pyglet.window.Window):
    '''
    '''

    def __init__(self, world, dt=1. / 30, paused=False, save_frames=None):
        # first, set up the pyglet screen, window, and display variables.
        platform = pyglet.window.get_platform()
        display = platform.get_default_display()
        screen = display.get_default_screen()
        try:
            config = screen.get_best_config(Config(
                alpha_size=8,
                depth_size=24,
                double_buffer=True,
                sample_buffers=1,
                samples=4))
        except pyglet.window.NoSuchConfigException:
            config = screen.get_best_config(Config())
        super(GL, self).__init__(resizable=True, config=config, vsync=False)

        # then, set up our own view parameters related to the simulator.
        self.dt = dt
        self.world = world
        self.paused = paused
        self.save_frames = save_frames
        self.frozen_bodies = []

        self.zoom = 10
        self.ty = 0.05
        self.tz = -0.8
        self.ry = 35
        self.rz = -60

        self.on_resize(self.width, self.height)

        glEnable(GL_BLEND)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_NORMALIZE)
        glEnable(GL_POLYGON_SMOOTH)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthFunc(GL_LEQUAL)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        glShadeModel(GL_SMOOTH)

        glLightfv(GL_LIGHT0, GL_AMBIENT, vec(0.2, 0.2, 0.2, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(1.0, 1.0, 1.0, 1.0))
        glLightfv(GL_LIGHT0, GL_POSITION, vec(3.0, 3.0, 10.0, 1.0))
        glEnable(GL_LIGHT0)

        self.box = build_vertex_list(*box_vertices())
        self.sphere = build_vertex_list(*sphere_vertices())
        self.cylinder = build_vertex_list(*cylinder_vertices(32))

    def _update_view(self):
        # http://njoubert.com/teaching/cs184_fa08/section/sec09_camera.pdf
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(1, 0, 0, 0, 0, 0, 0, 0, 1)
        glTranslatef(-self.zoom, self.ty, self.tz)
        glRotatef(self.ry, 0, 1, 0)
        glRotatef(self.rz, 0, 0, 1)

    def on_mouse_scroll(self, x, y, dx, dy):
        if dy == 0: return
        self.zoom *= 1.1 ** (-1 if dy > 0 else 1)
        self._update_view()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons == pyglet.window.mouse.LEFT:
            # pan
            self.ty += 0.03 * dx
            self.tz += 0.03 * dy
        else:
            # roll
            self.ry += 0.2 * -dy
            self.rz += 0.2 * dx
        self._update_view()

    def on_resize(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glu.gluPerspective(45, float(width) / height, 1, 100)
        self._update_view()

    def on_key_press(self, key, modifiers):
        keymap = pyglet.window.key
        if self.world.on_key_press(key, keymap):
            return
        if key == keymap.ESCAPE:
            pyglet.app.exit()
        if key == keymap.SPACE:
            self.paused = False if self.paused else True
        if key == keymap.ENTER:
            self.world.reset()
        if key == keymap.B:
            self.freeze_bodies()
        if key == keymap.F and self.save_frames:
            self.save_frame()
        if key == keymap.RIGHT:
            steps = int(1 / self.world.dt)
            if modifiers & keymap.MOD_SHIFT:
                steps *= 10
            [self.update(self.world.dt) for _ in range(steps)]

    def freeze_bodies(self):
        bodies = []
        for b in self.world.bodies:
            if b.name.startswith('table'):
                continue
            shape = {}
            if isinstance(b, physics.Sphere):
                shape = dict(radius=b.radius)
            if isinstance(b, physics.Box):
                shape = dict(lengths=b.lengths)
            if isinstance(b, physics.Cylinder):
                shape = dict(radius=b.radius, length=b.length)
            if isinstance(b, physics.Capsule):
                shape = dict(radius=b.radius, length=b.length)
            bp = b.__class__(b.name, self.world.ode_world, self.world.ode_space, color=b.color, **shape)
            bp.position = b.position
            bp.quaternion = b.quaternion
            bp.is_kinematic = True
            bodies.append(bp)
        self.frozen_bodies.append(bodies)

    def save_frame(self):
        bn = 'frame-{:05d}.png'.format(self.world.frame_no)
        fn = os.path.join(self.save_frames, bn)
        logging.info('saving frame %s', fn)
        pyglet.image.get_buffer_manager().get_color_buffer().save(fn)

    def redraw(self, dt):
        self.switch_to()
        self.clear()
        self.draw()
        self.flip()

    def update(self, dt):
        if self.paused:
            return
        if self.world.step():
            pyglet.app.exit()
        if self.world.needs_reset():
            self.world.reset()

    def run(self):
        pyglet.clock.schedule_interval(self.update, self.world.dt)
        pyglet.clock.schedule_interval(self.redraw, self.dt)
        pyglet.app.run()

    def draw(self):
        raise NotImplementedError


class Physics(GL):
    def __init__(self, *args, **kwargs):
        super(Physics, self).__init__(*args, **kwargs)
        BLK = [150, 150, 150] * 6
        WHT = [160, 160, 160] * 6
        N = 20
        z = kwargs.get('floor_z', 0)
        vtx = []
        for i in range(N, -N, -1):
            for j in range(-N, N, 1):
                vtx.extend((j,   i, z, j, i-1, z, j+1, i,   z,
                            j+1, i, z, j, i-1, z, j+1, i-1, z))

        self.floor = pyglet.graphics.vertex_list(
            len(vtx) // 3,
            ('v3f/static', vtx),
            ('c3B/static', ((BLK + WHT) * N + (WHT + BLK) * N) * N),
            ('n3i/static', [0, 0, 1] * (len(vtx) // 3)))

    def draw(self, color=None):
        '''Draw all bodies in the world.'''
        self.floor.draw(GL_TRIANGLES)
        for frame in self.frozen_bodies:
            for body in frame:
                self.draw_body(body)
        for body in self.world.bodies:
            self.draw_body(body, color=color)

    def draw_body(self, body, color=None):
        ''''''
        x, y, z = body.position
        r = body.rotation
        glColor4f(*(color if color is not None else body.color))
        with gl_context(mat=(r[0], r[3], r[6], 0.,
                             r[1], r[4], r[7], 0.,
                             r[2], r[5], r[8], 0.,
                             x, y, z, 1.)):
            if isinstance(body, physics.Box):
                x, y, z = body.lengths
                glScalef(x / 2., y / 2., z / 2.)
                self.box.draw(GL_TRIANGLES)
            elif isinstance(body, physics.Sphere):
                r = body.radius
                glScalef(r, r, r)
                self.sphere.draw(GL_TRIANGLES)
            elif isinstance(body, physics.Cylinder):
                l = body.length
                r = body.radius
                glScalef(r, r, l / 2)
                self.cylinder.draw(GL_TRIANGLES)
            elif isinstance(body, physics.Capsule):
                r = body.radius
                l = body.length
                with gl_context(scale=(r, r, l / 2)):
                    self.cylinder.draw(GL_TRIANGLES)
                with gl_context(mat=(r, 0, 0, 0,
                                     0, r, 0, 0,
                                     0, 0, r, 0,
                                     0, 0, -l / 2, 1)):
                    self.sphere.draw(GL_TRIANGLES)
                with gl_context(mat=(r, 0, 0, 0,
                                     0, r, 0, 0,
                                     0, 0, r, 0,
                                     0, 0, l / 2, 1)):
                    self.sphere.draw(GL_TRIANGLES)
