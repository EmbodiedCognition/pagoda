'''This module contains OpenGL code for rendering a world.'''

from __future__ import division

import contextlib
import logging
import numpy as np
import os
import pyglet

# normally this import should work; the try/except here is so the documentation
# will build on readthedocs.org!
try:
    from pyglet.gl import *
except ImportError:
    pass


@contextlib.contextmanager
def gl_context(scale=None, translate=None, rotate=None, mat=None, color=None):
    last_color = vec(0.0, 0.0, 0.0, 0.0)
    if color is not None:
        glGetFloatv(GL_CURRENT_COLOR, last_color)
        glColor4f(*color)
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
    if last_color is not None:
        glColor4f(*last_color)


def vec(*args):
    return (GLfloat * len(args))(*args)


def build_vertex_list(idx, vtx, nrm):
    return pyglet.graphics.vertex_list_indexed(
        len(vtx) // 3, idx, ('v3f/static', vtx), ('n3f/static', nrm))


def box_vertices():
    vtx = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, -1], [1, -1, 1],
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
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
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
    thetas = np.linspace(0, 2 * np.pi, n)
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


class View(object):
    '''A POD class for, in this case, holding view parameters.

    Any keyword arguments passed to the constructor will be set as attributes on
    the instance. This is used in the :class:`Window` class for holding
    parameters related to the view (i.e., zoom, translation, etc.).
    '''

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Window(pyglet.window.Window):
    '''This class wraps pyglet's Window for simple rendering of an OpenGL world.

    Default key bindings:
    - ESCAPE: close the window
    - SPACE: toggle pause
    - S: save a frame

    Parameters
    ----------
    paused : bool, optional
        Start the window with time paused. Defaults to False.
    floor_z : float, optional
        Height for a checkerboard floor in the rendered world. Defaults to 0.
        Set this to None to disable the floor.
    width : int, optional
        Initial width of the window. Defaults to 1027.
    height : int, optional
        Initial height of the window. Defaults to 603.

    Attributes
    ----------
    saved_frames : str
        Saved frames will be stored in this directory.
    paused : bool
        Current paused state of the renderer.
    frame_no : bool
        Number of the currently rendered frame, starting at 0. Increases by one
        with each call to :func:`render`.
    view : :class:`View`
        An object holding view parameters for the renderer.
    '''

    def __init__(self, dt=1. / 30, paused=False, save_frames=None, floor_z=0,
                 width=1200, height=675, resizable=True):
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

        super(Window, self).__init__(
            width=width, height=height, resizable=resizable, vsync=False, config=config)

        # then, set up our own view parameters.
        self.step_dt = self.render_dt = dt
        self.frame_no = 0
        self.paused = paused
        self.save_frames = save_frames
        self.view = View(zoom=4.666, ty=0.23, tz=-0.5, ry=27, rz=-50)

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

        self.floor = None
        if floor_z is not None:
            # set up a chessboard floor.
            BLK = [150, 150, 150] * 6
            WHT = [160, 160, 160] * 6
            N = 20
            z = floor_z
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

    def _update_view(self):
        # http://njoubert.com/teaching/cs184_fa08/section/sec09_camera.pdf
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(1, 0, 0, 0, 0, 0, 0, 0, 1)
        glTranslatef(-self.view.zoom, self.view.ty, self.view.tz)
        glRotatef(self.view.ry, 0, 1, 0)
        glRotatef(self.view.rz, 0, 0, 1)

    def on_mouse_scroll(self, x, y, dx, dy):
        if dy == 0:
            return
        self.view.zoom *= 1.1 ** (-1 if dy > 0 else 1)
        self._update_view()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons == pyglet.window.mouse.LEFT:
            # pan
            self.view.ty += 0.03 * dx
            self.view.tz += 0.03 * dy
        else:
            # roll
            self.view.ry += 0.2 * -dy
            self.view.rz += 0.2 * dx
        self._update_view()

    def on_resize(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glu.gluPerspective(45, float(width) / height, 1, 100)
        self._update_view()

    def on_key_press(self, key, modifiers):
        keymap = pyglet.window.key
        if self.grab_key_press(key, modifiers, keymap):
            return
        if key == keymap.ESCAPE:
            pyglet.app.exit()
        if key == keymap.SPACE:
            self.paused = False if self.paused else True
        if key == keymap.S and self.save_frames:
            self.save_frame()

    def save_frame(self, dt=None):
        bn = 'frame-{:05d}.png'.format(self.frame_no)
        fn = os.path.join(self.save_frames, bn)
        logging.info('saving frame %s', fn)
        pyglet.image.get_buffer_manager().get_color_buffer().save(fn)

    def _render(self, dt):
        self.switch_to()
        self.clear()
        if self.floor is not None:
            self.floor.draw(GL_TRIANGLES)
        self.render(dt)
        self.flip()

    def _step(self, dt):
        if not self.paused:
            self.frame_no += 1
            self.step(dt)

    def draw_sphere(self, *args, **kwargs):
        with gl_context(*args, **kwargs):
            self.sphere.draw(GL_TRIANGLES)

    def draw_box(self, *args, **kwargs):
        with gl_context(*args, **kwargs):
            self.box.draw(GL_TRIANGLES)

    def draw_cylinder(self, *args, **kwargs):
        with gl_context(*args, **kwargs):
            self.cylinder.draw(GL_TRIANGLES)

    def draw_lines(self, vertices, color=None):
        if color is not None:
            glColor4f(*color)
        glBegin(GL_LINES)
        for v in vertices:
            glVertex3f(*v)
        glEnd()

    set_color = glColor4f

    def exit(self):
        pyglet.app.exit()

    def run(self, movie=False):
        pyglet.clock.schedule_interval(self._step, self.step_dt)
        pyglet.clock.schedule_interval(self._render, self.render_dt)
        if movie and self.save_frames:
            pyglet.clock.schedule_interval(self.save_frame, self.render_dt)
        pyglet.app.run()

    def grab_key_press(self, key, modifiers, keymap):
        pass

    def step(self, dt):
        pass

    def render(self, dt):
        pass
