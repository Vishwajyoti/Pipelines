#IMAGE NAME WITH TAG TO BE RUN
IMAGE_NAME_WITH_TAG=$1
#PARAMETERS TO RUN THE IMAGE
#path
P=$2
#Target Varaible Name
T=$3

docker run -t ${IMAGE_NAME_WITH_TAG}  --path ${P} --target ${T}  
