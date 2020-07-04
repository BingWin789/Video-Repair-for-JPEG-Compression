#!/bin/bash

videodir=""
framedir=""
logsdir=""

for file in `ls $videodir`; do
    echo $file
    # parse file name
    tmpname="$(basename -- $file)"	
	sname=${tmpname##*-}
	bname=${sname%.*}

    log_name=$logsdir/$bname.log

    ffmpeg -i $videodir/$file -vsync 0 $framedir/${bname}%4d.png -y > ${log_name} 2>&1
done