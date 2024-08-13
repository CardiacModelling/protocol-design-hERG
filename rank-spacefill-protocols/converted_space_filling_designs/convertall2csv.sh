#!/bin/bash
FILES=*.eph
for f in $FILES
do
  echo "Processing $f file..."
  python ../../eph2csv.py $f
done
