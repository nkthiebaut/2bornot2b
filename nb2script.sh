#!/usr/bin/bash
# Convert a jupyter notebook to a python script file
# then remove all line containing 'print' except the last

nbfile=$1
scriptfile=${1%.ipynb}.py
echo "Output file: $scriptfile"

jupyter nbconvert --to script $nbfile
sed '/^$/d' $scriptfile |sed  '$ ! s/^.*print.*$//' > out.py
rm -f $scriptfile
