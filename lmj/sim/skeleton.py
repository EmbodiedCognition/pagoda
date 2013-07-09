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

'''ASF (skeleton file) parser.'''

from __future__ import print_function

import json
import lmj.cli
import numpy as np
import os
import re

from . import physics

logging = lmj.cli.get_logger(__name__)

TAU = 2 * np.pi


class ASF(object):
    def __init__(self):
        self.name = None
        self.version = None
        self.documentation = ''
        self.units = {}
        self.root = {}
        self.bones = {}
        self.hierarchy = {}

    @property
    def scale(self):
        '''Return a factor to convert length-scaled inches to mm.'''
        return 2.54 / (100. * self.units['length'])

    def yaml_lines(self):
        yield 'name: ' + self.name
        yield 'version: ' + self.version
        yield 'documentation: ' + self.documentation
        yield 'root:'
        for key, value in self.root.iteritems():
            if key in ('position', 'orientation', 'order'):
                yield '  %s:' % key
                for v in value:
                    yield '    - %s' % v
            else:
                yield '  %s: %s' % (key, value)
        yield 'units:'
        for key, value in self.units.iteritems():
            yield '  %s: %s' % (key, value)
        yield 'bones:'
        for bone in self.bones.itervalues():
            yield '  -'
            for line in bone.yaml_lines():
                yield '    ' + line
        yield 'hierarchy:'
        for key, values in self.hierarchy.iteritems():
            yield '  %s:' % key
            for v in values:
                yield '    - %s' % v

    def to_yaml(self):
        return '\n'.join(self.yaml_lines())

    def to_json(self):
        h = lambda x: x.__dict__ if isinstance(x, Bone) else x
        return json.dumps(self.__dict__, default=h)

    def create_bodies(self, world, translate=(0, 1, 0)):
        '''Traverse the bone hierarchy and create physics bodies.'''
        stack = [('root', 0, self.root['position'] + translate)]
        while stack:
            name, depth, end = stack.pop()

            for child in self.hierarchy.get(name, ()):
                b = self.bones[child]
                stack.append((child, depth + 1, end + b.direction * b.length))

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

    def create_joints(self, world):
        raise NotImplementedError


class Bone(object):
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

    def yaml_lines(self):
        yield 'id: %s' % self.id
        yield 'name: %s' % self.name
        yield 'direction:'
        for d in self.direction:
            yield '  - %s' % d
        yield 'length: %f' % self.length
        yield 'order: %s' % self.order
        yield 'axis:'
        for a in self.axis:
            yield '  - %f' % a
        yield 'dof:'
        for d in self.dof:
            yield '  - %s' % d
        yield 'limits:'
        for l, h in self.limits:
            yield '  -'
            yield '    low: %d' % l
            yield '    high: %d' % h

    def create_body(self, world, rank=0):
        return world.create_body('box',
                                 name=self.name,
                                 color=(rank, 0.7, 0.3),
                                 lengths=(0.05, 0.03, self.length))


class Tokenizer(list):
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
        print('Error at line %d, token %d: %r' % (i + 1, j + 1, t))

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
        logging.info('hierarchy: %s -> %s', source, ', '.join(targets))
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
        logging.info('bone %s: %dmm', bone.name, 1000 * bone.length)

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
                lo = float(tok.next().lstrip('('))
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

def parse(data):
    '''Parse an ASF skeleton definition file.

    Results are returned as an ASF object.
    '''
    asf = ASF()
    if os.path.exists(data):
        data = open(data)
    if isinstance(data, file):
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
        except Exception, e:
            tok.error()
            raise
        tok.end()
    return asf
