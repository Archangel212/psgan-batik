batik_dataset_name=4_Batik_500by500_homogenous_736-900
content_path=Single_Batik_729_500by500

#separate model hyperparameters and model state into two recursive directories
model_hyperparameters=kernel=5,zl_dim=40,zg_dim=20,zp_dim=3,ngf=128,ndf=128,batch_size=16
model_state=MLP,samefakeimg,G_upsampleConv2d,instance_noise_mean=0.1,shuffle_ds=False,real_label_smoothing=1,_test
experiment_name="$model_hyperparameters/$model_state"  

output_folder="$batik_dataset_name/GANOSAIC/content_path=$content_path/$experiment_name"
#make directory for batik dataset if it didn't exist in log directory 
mkdir -p ./log/$output_folder
echo $output_folder

# python PSGAN.py --texture_path batik_dataset/$batik_dataset_name --output_folder ./log/$batik_dataset_name/$experiment_name >> ./log/$batik_dataset_name/$model_hyperparameters/$model_state.txt

python GANOSAIC.py --texture_path batik_dataset/$batik_dataset_name --content_path batik_dataset/$content_path --test_image batik_dataset/Single_Batik_729_500by500/ --output_folder $output_folder  >> ./log/$output_folder/$model_state.txt

# #tail the log to get time lapse
cat ./log/$output_folder/$model_state.txt | tail -20 > "./log/$output_folder/$model_state.txt"
rm ./log/$output_folder/$model_state.txt

# #change log from txt to jpg format
soffice --convert-to jpg ./log/$output_folder/$model_state.txt --outdir ./log/$output_folder/

msg="Training $output_folder"

# git add --all
# git commit -m "$msg"
# git push -f

# echo "Finished" $msg

# sudo poweroff
