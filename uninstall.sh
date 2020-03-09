#!/bin/bash
name=`uname -a`
if [[ ${name} =~ "Darwin" ]]; then
	SH=zsh
	echo "mac"
elif [[ ${name} =~ "ubuntu" ]]; then
	SH=bash
	echo "ubuntu"
else
	SH=bash
	echo "linux"
fi

rm -rf ./libs
rm -rf ./build
rm -rf ./3rdparty/densecrf/build
cd ./python/utils
${SH} ./clean.sh
cd ../../
