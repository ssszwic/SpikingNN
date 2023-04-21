#!/bin/bash
# 杀死所有与train.py相关的进程
ps -ef | grep train.py | cut -c 12-16 | xargs kill -9