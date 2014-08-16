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

'''Base classes for simulations.'''


class World(object):
    '''World is a small base class for simulation worlds.'''

    def needs_reset(self):
        '''Return True iff the world needs to be reset.'''
        return False

    def reset(self):
        '''Reset the world state.'''
        pass

    def trace(self):
        '''Return a string containing world state for later analysis.'''
        pass

    def step(self):
        '''Advance the world simulation by one time step.'''
        raise NotImplementedError

    def on_key_press(self, key, keys):
        '''Handle an otherwise-unhandled keypress event.'''
        pass
