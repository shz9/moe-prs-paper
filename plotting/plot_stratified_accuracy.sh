#!/bin/bash

source env/moe/bin/activate

python plotting/plot_stratified_prediction_accuracy.py --phenotype LDL
python plotting/plot_stratified_prediction_accuracy.py --phenotype LDL --metric CORR
python plotting/plot_stratified_prediction_accuracy.py --phenotype HDL
python plotting/plot_stratified_prediction_accuracy.py --phenotype HDL --metric CORR
python plotting/plot_stratified_prediction_accuracy.py --phenotype LOG_TG
python plotting/plot_stratified_prediction_accuracy.py --phenotype LOG_TG --metric CORR
