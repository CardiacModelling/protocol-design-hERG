#!/bin/bash
FILES=*.txt
for f in $FILES
do
  echo "Processing $f file..."
  python ../../txt2eph.py $f
done
