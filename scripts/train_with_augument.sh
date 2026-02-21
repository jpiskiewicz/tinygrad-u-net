#!/bin/sh

# This script should only be used when we just generated the dataset
# and we run training on it for the first time. Otherwise running this
# is a waste of computing resources.
# It runs training and dataset augumentation in parallel.
# augument.py is churning along and spewing out new, augumented version
# of the original dataset and train.py is ingesting it.
# If augument.py isn't able to come up with the augumentation in time
# then the training routine just picks up the latest available augumentation.
# In this way the data augumentation becomes a non-blocking process.
# In the future I would like to save augumented datasets as files named as some hash
# so that they will only regenerate when there is no fresh version available.
 
export EPOCHS=200
AMD=1 BEAM=2 ./scripts/augument.py $EPOCHS &
aug_pid=$!
trap 'echo "Killing augument.py (pid $aug_pid)"; kill $aug_pid' EXIT
AMD=1 BEAM=2 ./scripts/train.py $EPOCHS