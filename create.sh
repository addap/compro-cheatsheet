#!/bin/bash

set -xe

cat $1/*.rs lib.rs > cheatsheet.txt
wc -c cheatsheet.txt