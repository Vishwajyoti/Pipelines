#Setting Gcloud credentials
gcloud auth activate-service-account --key-file=/home/vishwajyoti_pandey/component/Deployment/cloud-key/cohesive-gadget-166410-94640e49cd4c.json
#Name of the Bucket where model needs to be deployed
B_NAME=$1
#PROJECT ID OF THE PROJECT UNDER WHICH MODEL IS DEPLOYED
PROJECT_ID=$2
gcloud config set project ${PROJECT_ID}
#REGION UNDER WHICH MODEL IS DEPLOYED
REGION=$3
#APPENDING PROJECT NAME TO MAKE BUCKET NAME UNIQUE
BUCKET_NAME=${PROJECT_ID}-${B_NAME}
#CREATING A BUCKET FOR MODEL DEPLOYMENT 
gsutil mb -l ${REGION} gs://${BUCKET_NAME}
#MODEL PATH WHERE .pkl FILE IS STORED AND MODEL_NAME FOR ML-ENGINE
MODEL_PATH=$4
#COPYING THE .pkl FILE IN NEW BUCKET
gsutil cp  ${MODEL_PATH}models/model.pkl gs://${BUCKET_NAME}/
#GRANTING READ ACCESS TO MODEL FILE
gsutil acl ch -u AllUsers:R gs://${BUCKET_NAME}/model.pkl
MODEL_DIR=gs://${BUCKET_NAME}/
#FRAMEWORK & VERSION DETAILS
FRAMEWORK=$5
VAR=$(date +'%d%m%Y%H%M%S')
VERSION=$6${VAR}
MODEL_NAME=$7
#CREATING THE MODEL
gcloud ai-platform models create ${MODEL_NAME} --regions ${REGION}
#CREATING THE VERSION OF THE MODEL
gcloud ai-platform versions create $VERSION \
  --model $MODEL_NAME \
  --origin $MODEL_DIR \
  --runtime-version=1.14 \
  --framework $FRAMEWORK \
  --python-version=3.5 
