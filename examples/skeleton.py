#!/usr/bin/env python

import climate
import pagoda
import pagoda.viewer


@climate.annotate(
    asf='load skeleton data from this file',
    amc='load motion data from this file',
    )
def main(asf, amc):
    w = pagoda.skeleton.World(dt=1. / 60)
    w.add_motion(amc, name=w.add_skeleton(asf))
    pagoda.viewer.Viewer(w).run()


if __name__ == '__main__':
    climate.call(main)
