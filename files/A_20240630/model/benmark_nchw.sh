#!/bin/bash
benchmark_app -shape [1,3,256,256] -m nanonet-fp32-shape-1-3-256-256-model.xml -t 10
benchmark_app -shape [1,3,256,256] -m nanonet-fp32-shape-1-3-256-256-model.xml -t 10 -d GPU

