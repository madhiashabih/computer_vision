#!/bin/bash

# Create the q1/test_set and q1/train_set directories if they don't exist
mkdir -p q1/test_set
mkdir -p q1/train_set

# Loop through subjects s01 to s50
for i in $(seq -f "%02g" 1 50)
do
    # Generate a list of 5 random numbers between 1 and 15 for test set
    test_images=$(shuf -i 1-15 -n 5 | sort -n)
    
    # Move the randomly selected images for each subject to q1/test_set
    for num in $test_images
    do
        image_num=$(printf "%02d" $num)
        mv "q1/faces_transformed/s${i}_${image_num}.jpg" "q1/test_set/"
    done
    
    # Generate a list of 5 random numbers from the remaining 10 images for train set
    train_images=$(ls q1/faces_transformed/s${i}_*.jpg | shuf -n 5)
    
    # Move the randomly selected images for each subject to q1/train_set
    for image in $train_images
    do
        mv "$image" "q1/train_set/"
    done
    
    # The remaining 5 images stay in the q1/faces_transformed directory
done

# Print a message to confirm the operation is complete
echo "Finished moving random images for all subjects from s01 to s50."

# Count the number of images in the q1/test_set, q1/train_set, and remaining in q1/faces_transformed
test_count=$(ls -1 q1/test_set | wc -l)
train_count=$(ls -1 q1/train_set | wc -l)
remaining_count=$(ls -1 q1/faces_transformed | wc -l)
echo "Total images in q1/test_set directory: $test_count"
echo "Total images in q1/train_set directory: $train_count"
echo "Total images remaining in q1/faces_transformed directory: $remaining_count"
