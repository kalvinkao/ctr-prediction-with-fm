# Click Through Rate (CTR) Prediction using Factorization Machines at Scale

### The Team:
- [Colby Carter](https://github.com/colbycarter) (colby.carter@ischool.berkeley.edu)
- [Adam Letcher](https://github.com/perch333) (adam.letcher@ischool.berkeley.edu)
- [Jennifer Philippou](https://github.com/jphilippou27)
- [Kalvin Kao](https://github.com/kalvinkao) (kalvin.kao@ischool.berkeley.edu)

### Report
This project implements a scalable factorization machines approach for predicting click response to display advertising.  You can clone this repo to view the jupyter notebook, or you can view the PDF version (contains formatting differences) here:

[PDF Version](https://github.com/kalvinkao/ctr-prediction-with-fm/blob/master/CTR_FactorizationMachines.pdf)

An HTML version is also available here (but may not render due to size):

[HTML Version](https://github.com/kalvinkao/ctr-prediction-with-fm/blob/master/CarterKaoLetcherPhilippou_w261_FM_CTR.html)

### Instructions for Use
The jupyter notebook can be run, and requires a ./data folder containing the test.txt and train.txt files that are available from the link below.  If you are using docker, please use the compose file that is also available in this repo.

http://labs.criteo.com/2014/09/kaggle-contest-dataset-now-available-academic-use/


'submit_job_to_cluster.py' and 'fm_on_cluster.py' can also be used to submit a GCP dataproc job that runs training at scale, on a cluster.  You may be interested in setting the number of training iterations within 'fm_on_cluster.py'.  The other settings and tunable parameters are described in the report linked above.  Running on a dataproc cluster requires the following:

- Setting up and configuring your GCP account: see https://github.com/kalvinkao/ctr-prediction-with-fm/tree/master/GCP for instructions on how to do this (the instructions for creating an account apply to MIDS students only).
- Adding the data (linked above) to a /data folder within a GCP storage bucket.
- Creating a /results folder within the same GCP storage bucket.

Once your GCP account and storage bucket are ready, modify the two lines in 'fm_on_cluster.py' which point to the /data and /results folders within your bucket.  Then use the following command to submit a training job:

python submit_job_to_cluster.py --project_id=${PROJECT_ID} --zone==${ZONE} --cluster_name==${CLUSTER_NAME} --gcs_bucket=${BUCKET_NAME} --key_file=$HOME/KEYNAME.json --create_new_cluster --pyspark_file=fm_on_cluster.py --instance_type_m=${MASTER_MACHINE_TYPE} --instance_type_w=${WORKER_MACHINE_TYPE} --worker_nodes=${NUM_WORKER_NODES}

Here is an example job submission:

python submit_job_to_cluster.py --project_id=w261-223519 --zone=us-west1-a --cluster_name=cluster-jpalcckk-test03 --gcs_bucket=w261_jpalcckk --key_file=/home/muthderd/MIDS/w261.json --create_new_cluster --pyspark_file=fm_on_cluster.py --instance_type_m=n1-standard-8 --instance_type_w=n1-standard-4 --worker_nodes=16

Please be aware that you will be charged for using GCP storage and for submitting a GCP dataproc job!  Our team hopes you find this work interesting or useful, and please email us any questions, comments, or criticisms that you have-- we are always eager to learn and practice more!
