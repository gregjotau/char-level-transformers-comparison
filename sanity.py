#!/usr/bin/env python3
# coding: utf-8

# validation script to quickly check the Python environment.
# If there are any warnings or errors when running this, please seek
# assistance via the email address 'in5550-help@ifi.uio.no'.

import numpy as np
import matplotlib
import pandas as pd
import sklearn
import logging
import torch
import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print('Basic modules imported succcessfully!')

print('PyTorch version:', torch.__version__)

print('Success!\nYour environment seems to be healthy and suitable for IN5550.')
