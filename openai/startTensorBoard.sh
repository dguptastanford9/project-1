#!/bin/bash
tensorboard --logdir=run1:$1 --port 6006 > /dev/null 2>&1  
