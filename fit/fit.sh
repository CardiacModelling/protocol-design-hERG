#!/usr/bin/env bash

# Make sure logging folders exist
LOG="./log/fit-syn-opt-prt-hh_ikr_rt"
mkdir -p $LOG

echo "## fit syn." >> log/save_pid.log

# for cell in $CELLS  # or
for ((x=0; x<2; x++));
do
	echo "Cell $x"
	nohup python fit.py ../optimise/out/opt-prt-hh_ikr_rt.txt hh_ikr_rt 0 3 > $LOG/cell_$x.log 2>&1 &
	echo "# cell $x" >> log/save_pid.log
	echo $! >> log/save_pid.log
	sleep 5
done

