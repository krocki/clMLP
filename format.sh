#!/bin/bash
astyle --options=format_opts --recursive ./src/"*"
find ./src/ -name "*.orig" | xargs rm -rf