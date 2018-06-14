#!/usr/bin/env python

import click
import logging
import pagoda
import pagoda.viewer


@click.command()
@click.option('--asf', default='', help='load skeleton data from this file')
@click.option('--amc', default='', help='load motion data from this file')
def main(asf, amc):
    logging.basicConfig()
    w = pagoda.skeleton.World(dt=1. / 60)
    w.add_motion(amc, name=w.add_skeleton(asf))
    pagoda.viewer.Viewer(w).run()


if __name__ == '__main__':
    main()
