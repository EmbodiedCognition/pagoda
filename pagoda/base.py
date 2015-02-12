'''This module contains the base World class for simulations.'''


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
