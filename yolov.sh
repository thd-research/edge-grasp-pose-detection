#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# swap directories inside this script to ensure data paths are correct:
cd "$script_dir"

default="detected_parts"
default_dir="$script_dir"

# Set paths to Darknet files
darknet_path="darknetrun"
config_file="$script_dir/cfg/yolov4-custom.cfg"
#weights_file="$script_dir/weights/yolov4-custom_final.weights"
weights_file="$script_dir/weights/yolov4-custom_last.weights"
data_file="$script_dir/data/obj.data"
#threshold="0.3"


# get first and second ($1 and $2) argument from outside the script:
if [ $# -eq 0 ]; then # check if arguments are empty
	output_json=$default
	detect_dir=$default_dir
	echo  "Default directory chosen: $detect_dir"
	echo "Saving file as (basically as parts_<ts>) : $output_json.json"
else
	if [ -z "$2" ]; then #check if argument 2 is empty
		output_json=$1
		detect_dir=$default_dir
		echo "Default directory chosen: $detect_dir"
		echo "Save images as: $output_json.json"

	else
		output_json=$1
		detect_dir="$script_dir/../$2"
		echo "Get images and save output in dir: $detect_dir"
		echo "Save images as: $output_json.json"
	fi
fi


# normal command template is:
#darknetrun detector test data/obj.data cfg/yolov4-custom.cfg weights/yolov4-custom_5000.weights -ext_output -dont_show -out result.json images/captured_2023-10-27_1* > /dev/null 2>&1 &


# Directory containing images to process
image_dir="$detect_dir/images/"

# File to store the timestamp of the last iteration
timestamp_file="last_iteration_timestamp.txt"

# run darknet detection on the received image file:
#$darknet_path detector test $data_file $config_file $weights_file -ext_output -dont_show -out $detect_dir/$output_json.json $image_file > "detailed_detection_log" 2>&1


displayed=false # Set to display the last timestamp when detection performed

# if arguments are empty:
if [ $# -eq 0 ]; then
    # Get the timestamp of the last iteration
    last_timestamp=$(cat "$timestamp_file" 2>/dev/null)
   
    # Find the last modified image in the directory
    image_file=$(find "$image_dir" -type f -name "*.jpg" -newermt "$last_timestamp" | sort -n | tail -n 1)
    echo "$image_file"
    # Check if there's a new image to process
    if [ -n "$image_file" ]; then
        echo "Processing image: $image_file" | tee -a detection_log

        # Run YOLOv4 detection on the image
        $darknet_path detector test $data_file $config_file $weights_file -ext_output -dont_show -out $detect_dir/$output_json.json $image_file > "detailed_detection_log" 2>&1
	 #$darknet_path detector test $data_file $config_file $weights_file -ext_output -dont_show -out $output_json.json $image_file > /dev/null 2>&1 | tee -a "detection_log"


        # You can add additional processing or actions here if needed

        #echo "---------------------------------------"

        # Update the timestamp file with the current timestamp
        date +"%Y-%m-%d %H:%M:%S" > "$timestamp_file"
    	displayed=false
    else

	if [ "$displayed" = false ]; then
		last_detection_timestamp=$(cat "$timestamp_file" 2>/dev/null)
		echo "last image detection at: $last_detection_timestamp" | tee -a detection_log
		displayed=true
   	fi

    fi
        

    # Sleep for a specified interval before processing the images again
    #sleep 0.1
fi

