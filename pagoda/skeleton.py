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

'''ASF (skeleton file) and AMC (motion file) parsers.'''

import json
import climate
import numpy as np
import os
import re

from . import physics

logging = climate.get_logger(__name__)

TAU = 2 * np.pi


class World(physics.World):
    '''This world manages skeletons from ASF and motion from AMC files.'''

    def __init__(self, *args, **kwargs):
        super(World, self).__init__(*args, **kwargs)
        self.skeletons = {}
        self.motions = {}
        self.frames = {}
        self.frame = 0

    def add_skeleton(self, asf, name=None, translate=(0, 1, 0)):
        skeleton = parse_asf(asf)
        skeleton.create_bodies(self, translate=translate)
        #skeleton.create_joints(self)
        if name is None:
            name = skeleton.name
        self.skeletons[name] = skeleton
        self.motions[name] = None
        self.frames[name] = []
        return name

    def add_motion(self, amc, name=None):
        self.motions[name] = parse_amc(amc)

    def reset(self):
        self.frame = 0

    def step(self, substeps=2):
        '''Step the world forward by one frame.'''
        result = super(World, self).step(substeps=substeps)
        self.frame += 1
        for name in self.skeletons:
            if self.motions[name]:
                while self.frame >= len(self.frames[name]):
                    self.frames[name].append(next(self.motions[name]))
                self.skeletons[name].update_from_motion(self.frames[name][self.frame])
        return result


class Skeleton(object):
    '''This class handles configuration data from ASF files.'''

    def __init__(self):
        self.name = None
        self.version = None
        self.documentation = ''
        self.units = {}
        self.root = {}
        self.bones = {}
        self.hierarchy = {}

        self._bodies = {}

    @property
    def scale(self):
        '''Return a factor to convert length-scaled inches to mm.'''
        return 2.54 / (100. * self.units['length'])

    def to_json(self):
        h = lambda x: x.__dict__ if isinstance(x, Bone) else x
        return json.dumps(self.__dict__, default=h)

    def create_bodies(self, world, translate=(0, 1, 0)):
        '''Traverse the bone hierarchy and create physics bodies.'''
        stack = [('root', 0, self.root['position'] + translate)]
        while stack:
            name, depth, end = stack.pop()

            for child in self.hierarchy.get(name, ()):
                stack.append((child, depth + 1, end + self.bones[child].end))

            if name not in self.bones:
                continue

            bone = self.bones[name]
            body = bone.create_body(world, depth / 9.)

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

            self._bodies[name] = body

    def create_joints(self, world):
        '''Traverse the bone hierarchy and create physics joints.'''
        stack = ['root']
        while stack:
            name = stack.pop()
            for child in self.hierarchy.get(name, ()):
                stack.append(child)
            if name not in self.bones:
                continue
            bone = self.bones[name]
            body = self._bodies[name]
            for name in self.hierarchy.get(name, ()):
                child_bone = self.bones[name]
                child_body = self._bodies[name]
                shape = ('', 'hinge', 'universal', 'ball')[len(child_bone.dof)]
                world.join(shape, body, child_body)

    def update_from_motion(self, frame):
        pass


class Bone(object):
    '''A bone is an individual link in a skeleton.'''

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
        z = np.eye(3)
        for ax in self.order:
            theta = self.axis['XYZ'.index(ax)]
            ct = np.cos(theta)
            st = np.sin(theta)
            if ax == 'X':
                z = np.dot([[ 1,   0,  0], [ 0, ct, -st], [  0, st, ct]], z)
            if ax == 'Y':
                z = np.dot([[ct,   0, st], [ 0,  1,   0], [-st,  0, ct]], z)
            if ax == 'Z':
                z = np.dot([[ct, -st,  0], [st, ct,   0], [  0,  0,  1]], z)
        return z

    def create_body(self, world, rank=0):
        return world.create_body('box',
                                 name=self.name,
                                 color=tuple(np.random.rand(3)) + (0.9, ),#(rank, 0.7, 0.3, 0.9),
                                 lengths=(0.06, 0.03, self.length))


class Tokenizer(list):
    '''Tokenize a string by splitting on whitespace.

    Removes shell-style comments from the input. Keeps track of line and token
    number. Maintains some minimal state information for parsing nested blocks.
    '''

    class EOS(IndexError): pass
    class MissingEndKeyword(ValueError): pass

    def __init__(self, data):
        for i, l in enumerate(data.splitlines()):
            for j, t in enumerate(l.split('#')[0].strip().split()):
                self.append((i, j, t))
        self.index = 0
        self.begun = False

    @property
    def line_no(self):
        i, _, _ = self.peek(ahead=0, token_only=False)
        return i + 1

    def next(self, token_only=True):
        if self.index == len(self):
            raise Tokenizer.EOS()
        i, j, t = self[self.index]
        if self.begun and t.startswith(':'):
            raise Tokenizer.MissingEndKeyword()
        self.index += 1
        if token_only:
            return t
        return i, j, t

    def peek(self, ahead=0, token_only=True):
        if self.index + ahead >= len(self):
            return -1, -1, None
        i, j, t = self[self.index + ahead]
        if token_only:
            return t
        return i, j, t

    def error(self):
        i, j, t = self[self.index]
        return 'error at line %d, token %d: %r' % (i + 1, j + 1, t)

    def begin(self):
        self.begun = True

    def end(self):
        self.begun = False


def _parse_version(tok, asf):
    asf.version = tok.next()

def _parse_name(tok, asf):
    asf.name = tok.next()

def _parse_units(tok, asf):
    while not tok.peek().startswith(':'):
        key = tok.next()
        value = tok.next()
        if key in 'length mass':
            value = float(value)
        asf.units[key] = value

def _parse_documentation(tok, asf):
    doc = []
    while not tok.peek().startswith(':'):
        doc.append(tok.next())
    asf.documentation = ' '.join(doc)

def _parse_root(tok, asf):
    while not tok.peek().startswith(':'):
        key = tok.next()
        if key == 'position' or key == 'orientation':
            value = np.array([float(tok.next()) for _ in range(3)]) * asf.scale
        elif key == 'order':
            value = tuple(tok.next() for _ in range(6))
        else:
            value = tok.next()
        asf.root[key] = value

def _parse_hierarchy(tok, asf):
    assert tok.next() == 'begin'
    token = tok.next()
    while token != 'end':
        source = token
        line_no = tok.line_no
        targets = []
        while tok.line_no == line_no:
            targets.append(tok.next())
        asf.hierarchy[source] = tuple(targets)
        logging.debug('hierarchy: %s -> %s', source, ', '.join(targets))
        token = tok.next()

def _parse_bonedata(tok, asf):
    while not tok.peek().startswith(':'):
        bone = _parse_bone(tok)
        bone.length *= asf.scale
        # convert degrees to radians if needed.
        if asf.units['angle'].lower().startswith('deg'):
            bone.axis *= TAU / 360
            bone.limits *= TAU / 360
        asf.bones[bone.id] = asf.bones[bone.name] = bone
        logging.debug('bone %s: %dmm', bone.name, 1000 * bone.length)

def _parse_bone(tok):
    assert tok.next() == 'begin'
    bone = Bone()
    token = tok.next()
    while token != 'end':
        if token == 'id':
            bone.id = int(tok.next())
        if token == 'name':
            bone.name = tok.next()
        if token == 'direction':
            bone.direction = np.array([float(tok.next()) for _ in range(3)])
        if token == 'length':
            bone.length = float(tok.next())
        if token == 'axis':
            while re.match(r'^[-+eEgG.\d]+$', tok.peek()):
                bone.axis.append(float(tok.next()))
            bone.axis = np.array(bone.axis)
            if re.match(r'^[XYZ]+$', tok.peek().upper()):
                bone.order = tok.next().upper()
        if token == 'dof':
            while tok.peek() in 'rx ry rz':
                bone.dof.append(tok.next())
            bone.dof = tuple(bone.dof)
        if token == 'limits':
            while tok.peek().startswith('('):
                token = tok.next().lstrip('(').strip()
                lo = float(token) if token else float(tok.next())
                hi = float(tok.next().rstrip(')'))
                bone.limits.append((lo, hi))
        token = tok.next()
    bone.limits = np.array(bone.limits)
    return bone

PARSERS = dict(
    version=_parse_version,
    name=_parse_name,
    units=_parse_units,
    documentation=_parse_documentation,
    root=_parse_root,
    bonedata=_parse_bonedata,
    hierarchy=_parse_hierarchy,
    )

def parse_asf(data):
    '''Parse an ASF skeleton definition file.

    Results are returned as a Skeleton object.
    '''
    asf = Skeleton()
    if os.path.exists(data):
        logging.info('%s: loading skeleton data', data)
        data = open(data)
    if hasattr(data, 'read'):
        data = data.read()
    tok = Tokenizer(data)
    while True:
        token = None
        try:
            token = tok.next()
        except Tokenizer.EOS:
            break
        tok.begin()
        try:
            assert token.startswith(':')
            PARSERS[token[1:].lower()](tok, asf)
        except Exception as e:
            logging.critical(tok.error())
            raise
        tok.end()
    logging.info('parsed skeleton with %d bones', len(asf.bones))
    return asf


def parse_amc(data):
    '''Parse an AMC motion capture data file.

    Generates a sequence of frames. Each frame is a dictionary mapping a bone
    name to a list of the DOF configurations for that bone.
    '''
    if os.path.exists(data):
        logging.info('%s: loading motion data', data)
        data = open(data)
    if isinstance(data, str):
        data = data.splitlines()
    convert_degrees = False
    count = 0
    frame = {}
    for i, line in enumerate(data):
        line = line.split('#')[0].strip()
        if not line:
            continue
        try:
            if line.startswith(':'):
                assert count == 0
                line = line[1:].lower()
                if line.startswith('deg'):
                    convert_degrees = True
                continue
            if line.isdigit():
                if frame:
                    assert int(line) == count + 2
                    yield frame
                    count += 1
                    frame = {}
                continue
            name, dofs = line.split(None, 1)
            dofs = np.array(list(map(float, dofs.split())))
            if convert_degrees:
                dofs *= TAU / 360
            frame[name] = dofs
        except Exception as e:
            logging.critical('error at line %d, frame %d: %r', i + 1, count, line)
            raise
    logging.info('parsed %d frames of motion data', count)
