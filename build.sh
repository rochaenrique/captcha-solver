#!/bin/bash

if [ ! -d "build" ]; then
	mkdir build
fi

echoval() {
	echo $*
	eval $*
}

if [[ ! -d "data/test" || ! -d "data/train" ]]; then
    echo "NO DATA!!!!!"
    exit
fi
echo training!
SUBMIT="build/submit_$(date '+%d-%m_%H-%M-%S').csv"
echoval python3 model.py $SUBMIT

echo submiting!
echoval kaggle competitions submit -c yo-no-soy-un-bot -f $SUBMIT -m $(basename $SUBMIT)
