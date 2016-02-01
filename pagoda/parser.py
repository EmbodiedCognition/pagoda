'''Parser for configuring objects and joints in a pagoda simulation.'''

from __future__ import division, absolute_import, unicode_literals

import climate
import numpy as np
import os
import parsimonious
import re

from . import physics

logging = climate.get_logger(__name__)


def parse(source, world, jointgroup=None, density=1000, color=None):
    '''Load and parse a source file.

    Parameters
    ----------
    source : file
        A file-like object that contains text information describing bodies and
        joints to add to the world.
    world : :class:`pagoda.physics.World`
        The world to add objects and joints to.
    jointgroup : ode.JointGroup, optional
        If provided, add all joints from this parse to the given group. The
        default behavior adds joints to the world without an explicit group.
    density : float, optional
        Default density for bodies. This is overridden if the source provides a
        density or mass value for a given body.
    color : tuple of floats, optional
        Default color for bodies from this source. Defaults to None, which does
        not assign a color to parsed bodies.
    '''
    visitor = Visitor(world, jointgroup, density, color)
    visitor.parse(re.sub(r'#.*', ' ', source.read()))
    return visitor


class NodeVisitor(parsimonious.nodes.NodeVisitor):
    def generic_visit(self, node, children):
        return node.text

    def visit_(self, node, children):
        return children

    def visit_number(self, node, children):
        return float(node.text)

    def visit_float(self, node, children):
        return float(node.text)

    def visit_int(self, node, children):
        return int(node.text)


class Visitor(NodeVisitor):
    '''
    '''

    grammar = parsimonious.grammar.Grammar(
        open(os.path.join(os.path.dirname(__file__), 'pagoda.peg')).read())

    def __init__(self, world, jointgroup=None, density=1000, color=None,
                 *args, **kwargs):
        super(Visitor, self).__init__(*args, **kwargs)

        self.world = world
        self.jointgroup = jointgroup
        self.density = density
        self.color = color

        self.bodies = []
        self.joints = []

        self.attrs = {}

    def visit_tuplecolor(self, node, children):
        r, _, g, _, b, rest = tuple(children)
        a = rest[0][1] if rest else 1
        return r, g, b, a

    def visit_hexcolor(self, node, children):
        r = g = b = a = 1
        if 3 <= len(node.text) <= 4:
            r = int(node.text[0], 16) / 16
            g = int(node.text[1], 16) / 16
            b = int(node.text[2], 16) / 16
            if len(node.text) == 4:
                a = int(node.text[3], 16) / 16
        else:
            r = int(node.text[0:2], 16) / 256
            g = int(node.text[2:4], 16) / 256
            b = int(node.text[4:6], 16) / 256
            if len(node.text) == 8:
                a = int(node.text[6:8], 16) / 256
        return r, g, b, a

    def visit_color(self, node, children):
        self.attrs['color'] = children[-1][0]

    def visit_mass(self, node, children):
        self.attrs['mass'] = children[-1]

    def visit_density(self, node, children):
        self.attrs['density'] = children[-1]

    def visit_position(self, node, children):
        _, _, x, _, y, _, z = children
        self.attrs['position'] = x, y, z

    def visit_quaternion(self, node, children):
        _, _, theta, _, x, _, y, _, z = children
        self.attrs['quaternion'] = physics.make_quaternion(
            np.deg2rad(theta), x, y, z)

    def visit_handofgod(self, node, children):
        self.attrs['handofgod'] = True

    def visit_radius(self, node, children):
        self.attrs['radius'] = children[0]

    def visit_radlen(self, node, children):
        self.attrs['length'] = [x for x in children[0] if isinstance(x, float)][0]

    def visit_lengths(self, node, children):
        x, _, y, _, z = children
        self.attrs['lengths'] = x, y, z

    def visit_body(self, node, children):
        name, _, shape = children[:3]

        if 'mass' not in self.attrs and 'density' not in self.attrs:
            self.attrs['density'] = self.density

        logging.info('creating %s %s %s', shape, name, self.attrs)

        handofgod = self.attrs.pop('handofgod', None)
        position = self.attrs.pop('position', None)
        quaternion = self.attrs.pop('quaternion', None)
        color = self.attrs.pop('color', self.color)

        body = self.world.create_body(shape, name, **self.attrs)

        if quaternion:
            body.quaternion = quaternion
        if position:
            body.position = position
        if color:
            body.color = color
        if handofgod:
            logging.info('adding hand-of-god forces to %s', name)
            hog = physics.AMotor(
                '{}:hog'.format(name), self.world, body, mode='euler',
                jointgroup=self.jointgroup)
            hog.axes = [dict(rel=1, axis=(1, 0, 0)), None,
                        dict(rel=2, axis=(0, 0, 1))]
            self.joints.append(hog)

        self.bodies.append(body)

        self.attrs = {}

    def visit_attach(self, node, children):
        name, offset = children
        return name, offset[0][1] if offset else (0, 0, 0)

    def visit_offset(self, node, children):
        _, _, x, _, y, _, z, _, _ = children
        return x, y, z

    def visit_axis(self, node, children):
        _, _, x, _, y, _, z, _, _ = children
        return x, y, z

    def visit_range(self, node, children):
        lo, _, _, _, hi = children
        return lo, hi

    def visit_fmax(self, node, children):
        _, values = children
        self.attrs['fmax'] = [x for _, x in values]

    def visit_cfms(self, node, children):
        _, values = children
        self.attrs['cfms'] = [x for _, x in values]

    def visit_erps(self, node, children):
        _, values = children
        self.attrs['erps'] = [x for _, x in values]

    def visit_stops(self, node, children):
        _, ranges = children
        self.attrs['lo_stops'] = [a for _, (a, b) in ranges]
        self.attrs['hi_stops'] = [b for _, (a, b) in ranges]

    def visit_stop_cfms(self, node, children):
        _, values = children
        self.attrs['stop_cfms'] = [x for _, x in values]

    def visit_stop_erps(self, node, children):
        _, values = children
        self.attrs['stop_erps'] = [x for _, x in values]

    def visit_axes(self, node, children):
        _, axes = children
        self.attrs['axes'] = [ax for _, ax in axes]

    def visit_joint(self, node, children):
        shape, _, (body1, offset1), _, motion, _, (body2, offset2) = children[:7]

        if body1.lower() == 'world':
            body1 = None
        if body2.lower() == 'world':
            body2 = None

        anchor = None
        if motion == '<-':
            anchor = self.world.move_next_to(body1, body2, offset1, offset2)
        if motion == '->':
            anchor = self.world.move_next_to(body2, body1, offset2, offset1)
        if motion == '<>':
            anchor = self.world.get_body(body1).position  # TODO
        if shape.startswith('fix') or shape.startswith('sli'):
            anchor = None

        logging.info('joining %s <-%s-> %s at %s', body1, shape, body2,
                     [round(a, 3) for a in anchor or []])

        joint = self.world.join(shape, body1, body2, anchor=anchor,
                                jointgroup=self.jointgroup)

        default_axes = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        joint.axes = self.attrs.get('axes', default_axes[:joint.ADOF])

        if 'erps' in self.attrs:
            joint.erps = self.attrs['erps']
        if 'cfms' in self.attrs:
            joint.cfms = self.attrs['cfms']
        if 'fmax' in self.attrs:
            joint.fmax = self.attrs['fmax']
        if 'lo_stops' in self.attrs:
            if not shape.startswith('sli'):
                self.attrs['lo_stops'] = np.deg2rad(self.attrs['lo_stops'])
                self.attrs['hi_stops'] = np.deg2rad(self.attrs['hi_stops'])
            joint.lo_stops = self.attrs['lo_stops']
            joint.hi_stops = self.attrs['hi_stops']
            if 'stop_cfms' in self.attrs:
                joint.stop_cfms = self.attrs['stop_cfms']
            if 'stop_erps' in self.attrs:
                joint.stop_erps = self.attrs['stop_erps']

        self.joints.append(joint)

        self.attrs = {}

    def visit_top(self, node, children):
        mass = sum(b.mass.mass for b in self.bodies)
        vol = sum(b.volume for b in self.bodies)
        logging.info('%.1f mass / %f volume = %.1f overall density',
                     mass, vol, mass / vol)


def parse_asf(source, world, jointgroup=None, density=1000, color=None):
    '''Load and parse a source file.

    Parameters
    ----------
    source : file
        A file-like object that contains text information describing bodies and
        joints to add to the world.
    world : :class:`pagoda.physics.World`
        The world to add objects and joints to.
    jointgroup : ode.JointGroup, optional
        If provided, add all joints from this parse to the given group. The
        default behavior adds joints to the world without an explicit group.
    density : float, optional
        Default density for bodies. This is overridden if the source provides a
        density or mass value for a given body.
    color : tuple of floats, optional
        Default color for bodies from this source. Defaults to None, which does
        not assign a color to parsed bodies.
    '''
    visitor = AsfVisitor(world, jointgroup, density, color)
    visitor.parse(re.sub(r'#.*', ' ', source.read()))
    return visitor


class AsfBone(object):
    '''A bone is an individual link in an ASF-specified skeleton.'''

    def __init__(self):
        self.id = None
        self.name = ''
        self.direction = []
        self.length = -1
        self.axis = []
        self.order = 'XYZ'
        self.dof = []
        self.limits = []

    @property
    def end(self):
        return self.direction * self.length

    @property
    def rotation(self):
        m = np.eye(3)
        for ax in self.order:
            theta = self.axis['XYZ'.index(ax)]
            ct, st = np.cos(theta), np.sin(theta)
            if ax == 'X':
                m = np.dot([[1, 0, 0], [0, ct, -st], [0, st, ct]], m)
            if ax == 'Y':
                m = np.dot([[ct, 0, st], [0, 1, 0], [-st, 0, ct]], m)
            if ax == 'Z':
                m = np.dot([[ct, -st, 0], [st, ct, 0], [0, 0, 1]], m)
        return m


class AsfVisitor(NodeVisitor):
    '''Parse a skeleton definition in ASF format.

    For a description of the file format, see
    http://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html

    The format describes only the configuration of joints in an articulated
    body. It notably omits the dimensions and masses of the rigid bodies that
    connect the joints together, so we need additional information to construct
    a skeleton in the physics simulator.
    '''

    grammar = parsimonious.grammar.Grammar(
        open(os.path.join(os.path.dirname(__file__), 'asf.peg')).read())

    def __init__(self, world, jointgroup=None, density=1000, color=None,
                 *args, **kwargs):
        super(AsfVisitor, self).__init__(*args, **kwargs)

        self.world = world
        self.jointgroup = jointgroup
        self.density = density
        self.color = color

        self.version = ''
        self.name = ''
        self.units = {}
        self.root = {}
        self.documentation = ''

        self.bodies = []
        self.joints = []

        self.bone = AsfBone()
        self.bones = {}
        self.hierarchy = {}

    def visit_version(self, node, children):
        self.version = children[-1]

    def visit_asf_name(self, node, children):
        self.name = children[-1]

    def visit_documentation(self, node, children):
        self.documentation = ''.join(s + w for s, w in children[-1]).strip()

    def visit_mass_unit(self, node, children):
        self.units['mass'] = children[-1]

    def visit_length_unit(self, node, children):
        self.units['length'] = children[-1]

    def visit_angle_unit(self, node, children):
        self.units['angle'] = children[-1]

    def visit_root_axis(self, node, children):
        self.root['axis'] = children[-1][0]

    def visit_order(self, node, children):
        self.root['order'] = [dof for _, dof in children[-1]]

    def visit_position(self, node, children):
        _, _, x, _, y, _, z = children
        self.root['position'] = x, y, z

    def visit_orientation(self, node, children):
        _, _, x, _, y, _, z = children
        self.root['orientation'] = x, y, z

    def visit_bone(self, node, children):
        self.bones[self.bone.name] = self.bone
        self.bone = AsfBone()

    def visit_id(self, node, children):
        self.bone.id = children[-1]

    def visit_name(self, node, children):
        self.bone.name = children[-1]

    def visit_length(self, node, children):
        self.bone.length = children[-1]

    def visit_direction(self, node, children):
        _, _, x, _, y, _, z = children
        self.bone.direction = x, y, z

    def visit_axis(self, node, children):
        _, _, rx, _, ry, _, rz, _, order = children
        self.bone.axis = rx, ry, rz
        self.bone.order = order

    def visit_dofs(self, node, children):
        self.bone.dof = [dof for _, dof in children[-1]]

    def visit_limits(self, node, children):
        self.bone.limits = children[-1]

    def visit_limit(self, node, children):
        _, _, lo, _, hi, _, _ = children
        return lo, hi

    def visit_chain(self, node, children):
        parent, children, _ = children
        self.hierarchy[parent] = [id for _, id in children]

    @property
    def scale(self):
        '''Return a factor to convert length-scaled inches to mm.'''
        return 2.54 / (100. * self.units['length'])

    def create_bodies(self, translate=(0, 1, 0), size=0.1):
        '''Traverse the bone hierarchy and create physics bodies.'''
        stack = [('root', 0, self.root['position'] + translate)]
        while stack:
            name, depth, end = stack.pop()

            for child in self.hierarchy.get(name, ()):
                stack.append((child, depth + 1, end + self.bones[child].end))

            if name not in self.bones:
                continue

            bone = self.bones[name]
            body = self.world.create_body(
                'box', name=bone.name, density=density,
                lengths=(size, size, bone.length))

            # move the center of the body to the halfway point between
            # the parent (joint) and child (joint).
            x, y, z = end - bone.direction * bone.length / 2

            # swizzle y and z -- asf uses y as up, but we use z as up.
            body.position = x, z, y

            # compute an orthonormal (rotation) matrix using the ground and
            # the body. this is mind-bending but seems to work.
            u = bone.direction
            v = np.cross(u, [0, 1, 0])
            l = np.linalg.norm(v)
            if l > 0:
                v /= l
                rot = np.vstack([np.cross(u, v), v, u]).T
                swizzle = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
                body.rotation = np.dot(swizzle, rot).flatten()

            self.bodies.append(body)

    def create_joints(self):
        '''Traverse the bone hierarchy and create physics joints.'''
        stack = ['root']
        while stack:
            parent = stack.pop()
            for child in self.hierarchy.get(parent, ()):
                stack.append(child)
            if parent not in self.bones:
                continue
            bone = self.bones[parent]
            body = [b for b in self.bodies if b.name == parent][0]
            for child in self.hierarchy.get(parent, ()):
                child_bone = self.bones[child]
                child_body = [b for b in self.bodies if b.name == child][0]
                shape = ('', 'hinge', 'universal', 'ball')[len(child_bone.dof)]
                self.joints.append(self.world.join(shape, body, child_body))


def parse_amc(source):
    '''Parse an AMC motion capture data file.

    Parameters
    ----------
    source : file
        A file-like object that contains AMC motion capture text.

    Yields
    ------
    frame : dict
        Yields a series of motion capture frames. Each frame is a dictionary
        that maps a bone name to a list of the DOF configurations for that bone.
    '''
    lines = 0
    frames = 1
    frame = {}
    degrees = False
    for line in source:
        lines += 1
        line = line.split('#')[0].strip()
        if not line:
            continue
        if line.startswith(':'):
            if line.lower().startswith(':deg'):
                degrees = True
            continue
        if line.isdigit():
            if int(line) != frames:
                raise RuntimeError(
                    'frame mismatch on line {}: '
                    'produced {} but file claims {}'.format(lines, frames, line))
            yield frame
            frames += 1
            frame = {}
            continue
        fields = line.split()
        frame[fields[0]] = list(map(float, fields[1:]))
