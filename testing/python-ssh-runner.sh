#!/bin/bash

conda activate /data.nst/jfriedel/envs/neuro_op
cd /data.nst/jfriedel/projects/neuro_op/testing/
/data.nst/jfriedel/envs/neuro_op/bin/pip install -e ..
nohup /data.nst/jfriedel/envs/neuro_op/bin/python ./testruns.py >> log.txt 2> err.txt < /dev/null &