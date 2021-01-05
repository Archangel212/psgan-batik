batik_dataset_name=8_Batik_500by500_homogenous_736-900

model_hyperparameters=kernel=5,generator_upsampleConv2d,instance_noise_mean=0.1,zl_dim=40,zg_dim=20,zp_dim=3,ngf=128,ndf=128,learning_rate_g=2e-4,learning_rate_d=2e-4,batch_size=16,weight_decay=0
model_state=MLP_samefakeimg,No_Random_Cropping,Resize_dataset=160
experiment_name="$model_hyperparameters/$model_state"  

#make directory for batik dataset if it didn't exist in log directory 
mkdir -p ./log/$batik_dataset_name

python PSGAN.py --texture_path batik_dataset/$batik_dataset_name --output_folder ./log/$batik_dataset_name/$experiment_name >> ./log/$batik_dataset_name/$experiment_name.txt


#tail the log to get time lapse
cat ./log/$batik_dataset_name/$experiment_name.txt | tail -20 > "./log/$batik_dataset_name/$experiment_name/$experiment_name.txt"
rm ./log/$batik_dataset_name/$experiment_name.txt

#change log from txt to jpg format
soffice --convert-to jpg ./log/$batik_dataset_name/$experiment_name/$experiment_name.txt --outdir ./log/$batik_dataset_name/$experiment_name/

msg="Training ${batik_dataset_name} with ${experiment_name}"

git add --all
git commit -m "$msg"
git push -f

echo "Finished" $msg

sudo poweroff
