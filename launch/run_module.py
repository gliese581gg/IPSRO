#!/usr/bin/env python


import os
import sys


print sys.argv[1]
os.chdir(sys.argv[1])
os.system('python '+sys.argv[2])
