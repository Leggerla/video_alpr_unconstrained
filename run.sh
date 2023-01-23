#!/bin/bash

check_file() 
{
	if [ ! -f "$1" ]
	then
		return 0
	else
		return 1
	fi
}


# Check if Darknet is compiled
check_file "darknet/libdarknet.so"
retval=$?
if [ $retval -eq 0 ]
then
	echo "Darknet is not compiled! Go to 'darknet' directory and 'make'!"
	exit 1
fi

lp_model="data/lp-detector/wpod-net_update1.h5"
input_video=''
output_file=''
csv_file=''


# Check # of arguments
usage() {
	echo ""
	echo " Usage:"
	echo ""
	echo "   bash $0 -i input/dir -o output/dir -c csv_file.csv [-h] [-l path/to/model]:"
	echo ""
	echo "   -i   Input dir path (containing JPG or PNG images)"
	echo "   -o   Output dir path"
	echo "   -c   Output CSV file path"
	echo "   -l   Path to Keras LP detector model (default = $lp_model)"
	echo "   -h   Print this help information"
	echo ""
	exit 1
}

while getopts 'i:o:c:l:h' OPTION; do
	case $OPTION in
		i) input_video=$OPTARG;;
		o) output_file=$OPTARG;;
		l) lp_model=$OPTARG;;
		h) usage;;
	esac
done

if [ -z "$input_video"  ]; then echo "Input video  not set."; usage; exit 1; fi
if [ -z "$output_file" ]; then echo "Output file not set."; usage; exit 1; fi

# Check if input dir exists
check_file $input_video
retval=$?
if [ $retval -eq 0 ]
then
	echo "Input file ($input_video) does not exist"
	exit 1
fi

# Check if output dir exists, if not, create it
check_file $output_file
retval=$?
if [ $retval -eq 1 ]
then
	echo "Input file ($output_file) already exists"
	exit 1
fi

# End if any error occur
set -e

python3 run.py $input_video $output_file