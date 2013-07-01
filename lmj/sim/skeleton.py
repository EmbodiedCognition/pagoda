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

'''ASF parser.'''

import json


class ASF(object):
    def __init__(self):
        self.name = None
        self.version = None
        self.documentation = ''
        self.units = {}
        self.root = {}
        self.bones = {}
        self.hierarchy = {}

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


class Bone(object):
    def __init__(self):
        self.id = None
        self.name = ''
        self.direction = []
        self.length = -1
        self.axis = []
        self.dof = []
        self.limits = []

    def yaml_lines(self):
        yield 'id: %s' % self.id
        yield 'name: %s' % self.name
        yield 'direction:'
        for d in self.direction:
            yield '  - %s' % d
        yield 'length: %f' % self.length
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


class Tokenizer(object):
    class EOS(IndexError): pass
    class MissingEndKeyword(ValueError): pass

    def __init__(self, data):
        if isinstance(data, file):
            data = data.read()
        self.tokens = []
        for i, l in enumerate(data.splitlines()):
            for j, t in enumerate(l.split('#')[0].strip().split()):
                self.tokens.append((i, j, t))
        self.index = 0
        self.begun = False

    @property
    def line_no(self):
        i, _, _ = self.peek(ahead=0, token_only=False)
        return i + 1

    def next(self, token_only=True):
        if self.index == len(self.tokens):
            raise Tokenizer.EOS()
        i, j, t = self.tokens[self.index]
        if self.begun and t.startswith(':'):
            raise Tokenizer.MissingEndKeyword()
        self.index += 1
        if token_only:
            return t
        return i, j, t

    def peek(self, ahead=0, token_only=True):
        if self.index + ahead >= len(self.tokens):
            return -1, -1, None
        i, j, t = self.tokens[self.index + ahead]
        if token_only:
            return t
        return i, j, t

    def error(self, exc):
        i, j, t = self.tokens[self.index]
        print('Error at line %d, token %d: %r' % (i + 1, j + 1, t))
        raise exc

    def begin(self):
        self.begun = True

    def end(self):
        self.begun = False


def parse_version(tok, asf):
    asf.version = tok.next()

def parse_name(tok, asf):
    asf.name = tok.next()

def parse_units(tok, asf):
    while not tok.peek().startswith(':'):
        key = tok.next()
        asf.units[key] = tok.next()

def parse_documentation(tok, asf):
    doc = []
    while not tok.peek().startswith(':'):
        doc.append(tok.next())
    asf.documentation = ' '.join(doc)

def parse_root(tok, asf):
    while not tok.peek().startswith(':'):
        key = tok.next()
        if key == 'position' or key == 'orientation':
            value = tuple(float(tok.next()) for _ in range(3))
        elif key == 'order':
            value = tuple(tok.next() for _ in range(6))
        else:
            value = tok.next()
        asf.root[key] = value

def parse_hierarchy(tok, asf):
    assert tok.next() == 'begin'
    token = tok.next()
    while token != 'end':
        source = token
        line_no = tok.line_no
        targets = []
        while tok.line_no == line_no:
            targets.append(tok.next())
        asf.hierarchy[source] = tuple(targets)
        token = tok.next()

def parse_bonedata(tok, asf):
    while not tok.peek().startswith(':'):
        bone = parse_bone(tok, asf.root['axis'])
        asf.bones[bone.id] = asf.bones[bone.name] = bone

def parse_bone(tok, axis):
    bone = Bone()
    assert tok.next() == 'begin'
    token = tok.next()
    while token != 'end':
        if token == 'id':
            bone.id = int(tok.next())
        if token == 'name':
            bone.name = tok.next()
        if token == 'direction':
            bone.direction = tuple(float(tok.next()) for _ in range(3))
        if token == 'length':
            bone.length = float(tok.next())
        if token == 'axis':
            token = tok.next()
            while token != 'end' and token != axis:
                bone.axis.append(float(token))
                token = tok.next()
            bone.axis = tuple(bone.axis)
        if token == 'dof':
            while tok.peek() in 'rx ry rz':
                bone.dof.append(tok.next())
            bone.dof = tuple(bone.dof)
        if token == 'limits':
            while tok.peek().startswith('('):
                lo = float(tok.next().lstrip('('))
                hi = float(tok.next().rstrip(')'))
                bone.limits.append((lo, hi))
            bone.limits = tuple(bone.limits)
        token = tok.next()
    return bone

PARSERS = dict(
    version=parse_version,
    name=parse_name,
    units=parse_units,
    documentation=parse_documentation,
    root=parse_root,
    bonedata=parse_bonedata,
    hierarchy=parse_hierarchy,
    )

def parse(data):
    '''Parse an ASF skeleton definition file.

    Results are returned as an ASF object.
    '''
    asf = ASF()
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
            tok.error(e)
        tok.end()
    return asf
