'''This module contains OpenGL code for rendering a world.'''

from __future__ import division

import climate
import contextlib
import numpy as np
import os
import pyglet

from pyglet.gl import *

logging = climate.get_logger(__name__)

from . import physics

TAU = 2 * np.pi


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
    '''Pyglet event loop, prevents attribute changes from triggering redraws.'''

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

    def __init__(self, paused=False, floor_z=0, width=1072, height=603):
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
            width=width, height=height, resizable=True, vsync=False, config=config)

        # then, set up our own view parameters.
        self.frame_no = 0
        self.paused = paused
        self.saved_frames = None
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
        #print(self.view)

    def on_mouse_scroll(self, x, y, dx, dy):
        if dy == 0: return
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
        if key == keymap.S and self.saved_frames:
            self.save_frame()

    def save_frame(self, dt=None):
        if self.saved_frames is None:
            return
        bn = 'frame-{:05d}.png'.format(self.frame_no)
        fn = os.path.join(self.saved_frames, bn)
        logging.info('saving frame %s', fn)
        pyglet.image.get_buffer_manager().get_color_buffer().save(fn)

    def _render(self, dt):
        self.switch_to()
        self.clear()
        '''
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, 100, 0, 100)
        glMatrixMode(GL_MODELVIEW)
        pyglet.text.Label('Frame {}'.format(self.world.frame_no), x=0, y=0).draw()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        '''
        self.floor.draw(GL_TRIANGLES)
        self.render(dt)
        self.flip()

    def _step(self, dt):
        if not self.paused:
            self.frame_no += 1
            self.step(dt)

    def draw_sphere(self, *args, **kwargs):
        if 'color' in kwargs:
            glColor4f(*kwargs.pop('color'))
        with gl_context(*args, **kwargs):
            self.sphere.draw(GL_TRIANGLES)

    def draw_box(self, *args, **kwargs):
        if 'color' in kwargs:
            glColor4f(*kwargs.pop('color'))
        with gl_context(*args, **kwargs):
            self.box.draw(GL_TRIANGLES)

    def draw_cylinder(self, *args, **kwargs):
        if 'color' in kwargs:
            glColor4f(*kwargs.pop('color'))
        with gl_context(*args, **kwargs):
            self.cylinder.draw(GL_TRIANGLES)

    def draw_lines(self, vertices):
        glBegin(GL_LINES)
        for v in vertices:
            glVertex3f(*v)
        glEnd()

    set_color = glColor4f

    def exit(self):
        pyglet.app.exit()

    def run(self, step_dt=1 / 30, render_dt=1 / 30, movie=None):
        '''Run the pyglet window.

        Parameters
        ----------
        step_dt : float, optional
            Time interval between successive calls to :func:`step`. Defaults to
            0.033333 (30 fps).
        render_dt : float, optional
            Time interval between successive calls to :func:`render`. Defaults
            to 0.033333 (30 fps).
        movie : str, optional
            If given, save rendered frames to images in this directory.
        '''
        pyglet.clock.schedule_interval(self._step, step_dt)
        pyglet.clock.schedule_interval(self._render, render_dt)
        if movie is not None:
            self.saved_frames = movie
            pyglet.clock.schedule_interval(self.save_frame, render_dt)
        pyglet.app.run()

    def grab_key_press(self, key, modifiers, keymap):
        pass

    def step(self, dt):
        pass

    def render(self, dt):
        pass


class Null(object):
    '''This viewer does nothing!

    It is here for headless simulation runs, i.e., where the world does not need
    to be rendered to a graphics window but some measurements or other
    side-effects of repeatedly stepping the world are useful.
    '''

    def __init__(self, world):
        self.world = world

    def run(self):
        while not self.world.step():
            if self.world.needs_reset():
                self.world.reset()


class Viewer(Window):
    '''A viewer window for pagoda worlds.

    This viewer adds the following default keybindings:
    - F: freeze state of current bodies in the world
    - RIGHT-ARROW: advance world state by 1 frame (10 if SHIFT)
    - ENTER: reset the world

    Parameters
    ----------
    world : :class:`pagoda.World`
        A world object to view in this window.

    Attributes
    ----------
    world : :class:`pagoda.World`
        The world object being viewed in this window.
    '''

    def __init__(self, world, *args, **kwargs):
        super(Viewer, self).__init__(*args, **kwargs)
        self.world = world
        self._frozen = []

    def grab_key_press(self, key, modifiers, keymap):
        if key == keymap.ENTER:
            self.world.reset()
            return True
        if key == keymap.F:
            self.freeze_bodies()
            return True
        if key == keymap.RIGHT:
            steps = int(1 / self.world.dt)
            if modifiers & keymap.MOD_SHIFT:
                steps *= 10
            [self.step(self.world.dt) for _ in range(steps)]
            return True

    def run(self, render_dt=1 / 30, movie=None):
        super(Viewer, self).run(
            step_dt=self.world.dt, render_dt=1 / 60, movie=movie)

    def step(self, dt):
        if self.world.step():
            self.exit()
        if self.world.needs_reset():
            self.world.reset()

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
        self._frozen.append(bodies)

    def render(self, dt):
        '''Draw all bodies in the world.'''
        for frame in self._frozen:
            for body in frame:
                self.draw_body(body)
        for body in self.world.bodies:
            self.draw_body(body)

        if hasattr(self.world, 'markers'):
            # draw line between anchor1 and anchor2 for marker joints.
            glColor4f(0.9, 0.1, 0.1, 0.9)
            glLineWidth(3)
            for j in self.world.markers.joints:
                glBegin(GL_LINES)
                glVertex3f(*j.getAnchor())
                glVertex3f(*j.getAnchor2())
                glEnd()

    def draw_body(self, body):
        ''''''
        x, y, z = body.position
        r = body.rotation
        glColor4f(*body.color)
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
