#!/usr/bin/env python

import climate
import os
import pagoda
import pagoda.parser
import pagoda.physics
import pagoda.viewer


class Viewer(pagoda.viewer.Viewer):
    def grab_key_press(self, key, modifiers, keymap):
        if key == keymap.B:
            self.world.get_body('arm').add_force(
                (0, 0, 100), relative_position=(1, 0, 0))
            return True


def main():
    w = pagoda.physics.World(dt=0.01)
    with open(os.path.join(os.path.dirname(__file__), 'hinge-limits.txt')) as h:
        pagoda.parser.parse(h, w)
    w.get_body('arm').add_force((0, 0, 100), relative_position=(1, 0, 0))
    Viewer(w).run()


if __name__ == '__main__':
    climate.call(main)
