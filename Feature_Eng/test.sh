#IMAGE NAME WITH TAG TO BE RUN
IMAGE_NAME_WITH_TAG=$1
#PARAMETERS TO RUN THE IMAGE
#path
P=$2
#Filename
F=$3
#Test Size
T=$4

docker run -t ${IMAGE_NAME_WITH_TAG}  --path ${P} --filename ${F} --t_size ${T} 
