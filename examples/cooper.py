#!/usr/bin/env python

import climate
import os
import pagoda
import pagoda.viewer
import sys

def full(name):
    return os.path.join(os.path.dirname(__file__), name)

def main():
    w = pagoda.cooper.World(dt=1. / 120)
    w.erp = 0.3
    w.cfm = 1e-6
    w.load_skeleton(full('cooper-skeleton.txt'))
    w.load_markers(full('cooper-motion.c3d'), full('cooper-markers.txt'))
    pagoda.viewer.Viewer(w).run()


if __name__ == '__main__':
    climate.call(main)
