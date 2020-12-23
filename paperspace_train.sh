batik_dataset_name=Single_Batik_70_500by500
experiment_name=kernel=5,generator_upsampleConv2d,zl_dim=40,zg_dim=20,_dim=3,learning_rate_g=2e-4,learning_rate_d=2e-4
# experiment_name=kernel=4,generator_leakyRelu=0.2,instance_noise_mean=0_std=0.1,label_smoothing=0.0955percent,learning_rate_g=1e-4,learning_rate_d=4e-4

python PSGAN.py --texture_path batik_dataset/$batik_dataset_name --output_folder ./log/$batik_dataset_name/$experiment_name >> ./log/$batik_dataset_name/$experiment_name.txt


#tail the log to get time lapse
cat ./log/$batik_dataset_name/$experiment_name.txt | tail -20 > "./log/$batik_dataset_name/$experiment_name/$experiment_name.txt"
rm ./log/$batik_dataset_name/$experiment_name.txt

#change log from txt to jpg format
soffice --convert-to jpg ./log/$batik_dataset_name/$experiment_name/$experiment_name.txt --outdir ./log/$batik_dataset_name/$experiment_name/

msg="Training ${batik_dataset_name} with ${experiment_name}"

#git add --all
#git commit -m "$msg"
#git push 

echo "Finished" $msg

#sudo poweroff
