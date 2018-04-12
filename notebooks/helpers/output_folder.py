import os
import sys
import datetime

def output_folder(prefix='output-'):
    relpath = os.path.join(os.path.dirname(__file__), '..', 'logs')
    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    f = '{}{}'.format(prefix, ts)
    return os.path.abspath(os.path.join(relpath, f))
