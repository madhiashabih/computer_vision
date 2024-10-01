#!/bin/bash

# Create the faces_test directory if it doesn't exist
mkdir -p faces_test

# Loop through subjects s03 to s50
for i in $(seq -f "%02g" 3 50)
do
    # Move the specified images for each subject
    mv faces_transformed/s${i}_03.jpg \
       faces_transformed/s${i}_06.jpg \
       faces_transformed/s${i}_08.jpg \
       faces_transformed/s${i}_11.jpg \
       faces_transformed/s${i}_15.jpg \
       faces_test/

    echo "Moved 5 images for subject s${i} to faces_test directory."
done

# Print a message to confirm the operation is complete
echo "Finished moving test images for all subjects from s03 to s50."

# Optional: Count the number of images in the faces_test directory
image_count=$(ls -1 faces_test | wc -l)
echo "Total images in faces_test directory: $image_count"