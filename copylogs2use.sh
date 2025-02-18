#!/bin/bash
# copy_script.sh
# This script copies the directory specified by combining various path parts

# Define the base source and destination directories
SOURCE="/data/junseolee/diffuser/intern/"
DESTINATION="/home/junseolee/24WIN_RLLAB_INTERN/"

# Define the additional path components
#ADD1="diffuser/logs/halfcheetah-medium-expert-v2/"
ADD1="diffuser/logs/halfcheetah-medium-replay-v2/"
ADD2="values/diffusion_H4_T1_S0_d0.99/"
#ADD2="diffusion/diffuser_H4_T1_S0"
#ADD2="diffusion/diffuser_H4_T1_S0"


# Construct the full source and destination paths
FULL_SOURCE="${SOURCE}${ADD1}${ADD2}"
FULL_DESTINATION="${DESTINATION}${ADD1}${ADD2}"

# Display the constructed paths
echo "Copying from:"
echo "  $FULL_SOURCE"
echo "to:"
echo "  $FULL_DESTINATION"

# Copy the directory recursively
cp -r "$FULL_SOURCE" "$FULL_DESTINATION"

# Check if the copy was successful
if [ $? -eq 0 ]; then
    echo "Copy succeeded."
else
    echo "Copy failed." >&2
fi
