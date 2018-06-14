#!/usr/bin/env python

import click
import logging
import os
import pagoda
import pagoda.viewer


def full(name):
    return os.path.join(os.path.dirname(__file__), name)


@click.command()
def main():
    logging.basicConfig()
    w = pagoda.cooper.World(dt=1. / 120)
    w.load_skeleton(full('../optimized-skeleton.txt'))
    w.load_markers(full('cooper-motion.c3d'), full('../optimized-markers.txt'))
    pagoda.viewer.Viewer(w).run()


if __name__ == '__main__':
    main()
