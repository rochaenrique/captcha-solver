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

find_python() {
	# try python3
	if [[ -n $(which python3.9) ]]; then
		echo "python3.9"
	elif [[ -n $(which python3) ]]; then
	   PY_VER=$(python3 --version | cut -d' ' -f2)
	   if [[ $PY_VER =~ ^3\.9\.[0-9]{1,2}$ ]]; then
		   echo "python3"
	   fi
	fi
}

validate_python() {
	PYTHON=$(find_python)
	if [[ -z $PYTHON ]]; then
		echo could not find python3.9!
		exit 1
	fi
}

if [[ $1 = "env" ]]; then
	if [[ ! -d .venv ]]; then
		validate_python
		echoval $PYTHON -m venv .venv
	fi
	if [[ -z $(which pip) ]]; then
		echo could not find pip! activate env!
		exit 1
	fi
	if [[ -e requirements.txt ]]; then
		echoval pip install -r requirements.txt
	else
		echo no requirements.txt! get it!
		exit 1
	fi

elif [[ $1 = "download" ]]; then
	echo downloading!
	if [[ ! -d data ]]; then
		mkdir data
	fi
	if [[ -z $(which kaggle) ]]; then
		echo no kaggle! install and start env!
		exit 1
	fi
	echoval kaggle competitions download -c yo-no-soy-un-bot -p data
	echoval cd data
	echoval unzip *
else
	validate_python
	echo training!
	SUBMIT="build/submit_$(date '+%d-%m_%H-%M-%S').csv"
	echoval python3 model.py $SUBMIT

	if [[ $? && $1 = "submit" ]]; then
		echo submiting!
		echoval kaggle competitions submit -c yo-no-soy-un-bot -f $SUBMIT -m $(basename $SUBMIT)
	fi
fi

