#!/bin/bash

if [ -z "$*" ]; then
	echo "Usage: [submit]"
fi

if [ ! -d "build" ]; then
	mkdir build
fi

echoval() {
	echo $*
	eval $*
}

if [[ $1 = "download" ]]; then
	echo downloading!
	if [[ ! -d data ]]; then
		mkdir data
	fi
	echo kaggle competitions download -c yo-no-soy-un-bot -p data
	cd data
	unzip *
else
	echo training!
	SUBMIT="build/submit_$(date '+%d-%m_%H-%M-%S').csv"
	echoval python3 model.py $SUBMIT

	if [[ $? && $1 = "submit" ]]; then
		echo submiting!
		echoval kaggle competitions submit -c yo-no-soy-un-bot -f $SUBMIT -m $(basename $SUBMIT)
	fi
fi

