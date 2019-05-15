#!/usr/bin/env bash

PRTID="staircaseramp"
EXPID="newtonrun1"
INFOID="hh_ikr_rt"

# Make sure logging folders exist
LOG="./log/fit-$FILENAME"
mkdir -p $LOG

# (.) turns grep return into array
# use grep with option -e (regexp) to remove '#' starting comments
CELLS=(`grep -v -e '^#.*' ./data/selected-$FILENAME.txt`)

echo "## fit-$FILENAME" >> log/save_pid.log

# for cell in $CELLS  # or
for ((x=0; x<2; x++));
do
	echo "${CELLS[x]}"
	nohup python fit.py $PRTID $EXPID ${CELLS[x]} $INFOID 10 > $LOG/${CELLS[x]}.log 2>&1 &
	echo "# ${CELLS[x]}" >> log/save_pid.log
	echo $! >> log/save_pid.log
	sleep 5
done
