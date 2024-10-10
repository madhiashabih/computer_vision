#!/bin/bash

# Create the q1/test_set and q1/train_set directories if they don't exist
mkdir -p q1/test_set
mkdir -p q1/train_set_1

for i in $(seq -f "%02g" 1 50)  # Loop through subjects 01 to 50
do
    # Move images 1-5 to the test set
    for num in $(seq -f "%02g" 1 5)  # Images 01, 02, 03, 04, 05
    do
        mv "q1/train_set_2/s${i}_${num}.jpg" "q1/test_set/"
    done

    # Move images 6-10 to the train set
    for num in $(seq -f "%02g" 6 10)  # Images 06, 07, 08, 09, 10
    do
        mv "q1/train_set_2/s${i}_${num}.jpg" "q1/train_set_1/"
    done

    # The remaining images (11-15) stay in the `q1/train_set_2` directory
done

# Print a message to confirm the operation is complete
echo "Finished moving random images for all subjects from s01 to s50."

# Count the number of images in the q1/test_set, q1/train_set, and remaining in q1/train_set_2
test_count=$(ls -1 q1/test_set | wc -l)
train_count=$(ls -1 q1/train_set_1 | wc -l)
remaining_count=$(ls -1 q1/train_set_2 | wc -l)
echo "Total images in q1/test_set directory: $test_count"
echo "Total images in q1/train_set_1 directory: $train_count"
echo "Total images remaining in q1/train_set_2 directory: $remaining_count"
