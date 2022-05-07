DATASET=viggo
TRAINING_DATA_FOLDER=../data/${DATASET}_consistency/
OUTPUT_FOLDER=${HOME}/datatuner/trained_classifiers/${DATASET}
TRAINING_ARGS=./classifier_training_args/${DATASET}/${DATASET}_model_training_args.json
DATE=$(date +%s)
LOG_OUTPUT_PATH=./outputs/classifier_train_${DATASET}_${DATE}.out

export CUDA_VISIBLE_DEVICES="0,2"

echo "Starting classifier training, reading data from $TRAINING_DATA_FOLDER and outputting the model at $OUTPUT_FOLDER"
echo "Saving output log at $LOG_OUTPUT_PATH"

bash train_classifier.sh $TRAINING_DATA_FOLDER $OUTPUT_FOLDER $TRAINING_ARGS 2 &> $LOG_OUTPUT_PATH &
tail -f $LOG_OUTPUT_PATH
