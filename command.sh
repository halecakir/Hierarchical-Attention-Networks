#!/bin/bash

PERMISSION_TYPE=$1
MODEL_TYPE="HAN"
OUTPUT_DIR="output/reports/$MODEL_TYPE"
PARAMETERS_DIR="/home/huseyinalecakir/HAN/data/saved-parameters"
# Create output directory if not exists
mkdir -p $OUTPUT_DIR
rm -f $OUTPUT_DIR/$MODEL_TYPE-$PERMISSION_TYPE.out

python runner.py 	--permission-type $PERMISSION_TYPE \
					--saved-data $PARAMETERS_DIR/saved-data/emdeddings-documents-with-g5-w2i.pickle \
					--saved-reviews $PARAMETERS_DIR/saved-data/reviews.pickle \
					--saved-predicted-reviews $PARAMETERS_DIR/saved-data/predicted-$PERMISSION_TYPE-reviews.pickle \
					--model-checkpoint $PARAMETERS_DIR/saved-models/$MODEL_TYPE-$PERMISSION_TYPE.pt \
					--outdir $OUTPUT_DIR/$MODEL_TYPE-$PERMISSION_TYPE.out
