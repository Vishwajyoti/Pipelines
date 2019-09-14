"""
Created on Tue Sep  3 14:58:04 2019

@author: vispande2
"""


from kfp import compiler
import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.notebook
import sys
import json

class ObjectDict(dict):
  def __getattr__(self, name):
    if name in self:
      return self[name]
    else:
      raise AttributeError("No such attribute: " + name)


@dsl.pipeline(
  name='Basic KubeFlow Pipeline',
  description='Feature Eng,Training,Testing & Deployment'
)
def ftreng_train_test_and_deploy(
        project='cohesive-gadget-166410',
        bucket_uri='gs://vishwa/',
        region='us-central1',
        test_size=0.3,
        file_name='Boston.csv',
        target_var='target',
        hyper_param=json.dumps({'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],'max_features': ['auto', 'sqrt'],'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4],'bootstrap': [True, False]}),
        search_type=1,
        model_bucket_name='rf-vj-model',
        model_name='rf-vj',
        framework='scikit-learn',
        version_key_word='rf_model_'
        
        ):
# Step 1: create training dataset using Apache Beam on Cloud Dataflow
    feature_eng = dsl.ContainerOp(
            name='feature_eng',
            # image needs to be a compile-time string
            image='gcr.io/cohesive-gadget-166410/feature-eng-vj:latest',
            arguments=[
                    '--path',bucket_uri,
                    '--filename',file_name,
                    '--t_size',test_size
                    ]
            #,file_outputs={'bucket': '/output.txt'}
            ).apply(gcp.use_gcp_secret('user-gcp-sa'))

# Step 2: Train the model and find best set of hyperparameter.
    train = dsl.ContainerOp(
            name='train',
            # image needs to be a compile-time string
            image='gcr.io/cohesive-gadget-166410/train-rf-vj:latest',
            arguments=[
                    '--path',bucket_uri,
                    '--target',target_var,
                    '--h_param',hyper_param,
                    '--search_type',search_type
                    ]
            #,file_outputs={'jobname': '/output.txt'}
            ).apply(gcp.use_gcp_secret('user-gcp-sa'))
    train.after(feature_eng)

# Step 3: Train the model some more, but on the pipelines cluster itself
    test = dsl.ContainerOp(
            name='test',
            # image needs to be a compile-time string
            image='gcr.io/cohesive-gadget-166410/test-rf-vj:latest',
            #image='gcr.io/cloud-training-demos/babyweight-pipeline-traintuned-trainer@sha256:3d73c805430a16d0675aeafa9819d6d2cfbad0f0f34cff5fb9ed4e24493bc9a8',
            arguments=[
                    '--path',bucket_uri,
                    '--target',target_var
                    ]
            #,file_outputs={'train': '/output.txt'}
            ).apply(gcp.use_gcp_secret('user-gcp-sa'))
    test.after(train)

# Step 4: Deploy the trained model to Cloud ML Engine
    deploy_cmle = dsl.ContainerOp(
            name='deploycmle',
            # image needs to be a compile-time string
            image='gcr.io/cohesive-gadget-166410/deploy-rf-vj:latest',
            arguments=[
                    model_bucket_name,
                    project,
                    region,
                    bucket_uri,
                    framework,
                    version_key_word,
                    model_name   
                    ]

        ).apply(gcp.use_gcp_secret('user-gcp-sa'))
    deploy_cmle.after(test)

if __name__ == '__main__':
    filename=sys.argv[1]
    compiler.Compiler().compile(ftreng_train_test_and_deploy,filename)
