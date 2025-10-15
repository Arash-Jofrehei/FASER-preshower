#!/bin/bash

inFile=$1
run=${inFile:31:6}

mkdir -p /eos/user/a/ajofrehe/FASER/preshower/decoded/$run
cd /eos/user/a/ajofrehe/FASER/preshower/decoded/$run
/afs/cern.ch/work/a/ajofrehe/FASER/preshower/faser-common/build/EventFormats/decodeHalfPlaneFaserDAQ --boardIDs 0,1,2,3,4,5,6,7 $inFile
cd -