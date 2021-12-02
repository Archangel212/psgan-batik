training()
{
    batik_dataset_name=$1

    #separate model hyperparameters and model state into two recursive directories
    model_hyperparameters=$2
    model_state=$3
    experiment_name="$model_hyperparameters/$model_state"  
    
    #make directory for batik dataset if it didn't exist in log directory 
    mkdir -p ./log/$batik_dataset_name/$experiment_name

    python PSGAN.py --texture_path batik_dataset/$batik_dataset_name --output_folder ./log/$batik_dataset_name/$experiment_name --nBlocksG $4 >> ./log/$batik_dataset_name/$model_hyperparameters/$model_state.txt


    #compute density and coverage for recently trained model 
    python metrics.py --texture_path batik_dataset/$batik_dataset_name --output_folder ./log/$batik_dataset_name/$experiment_name --nBlocksG $4 >> ./log/$batik_dataset_name/$model_hyperparameters/$model_state.txt


    #tail the log to get time lapse
    cat ./log/$batik_dataset_name/$model_hyperparameters/$model_state.txt | tail -20 > "./log/$batik_dataset_name/$experiment_name/$model_state.txt"
    rm ./log/$batik_dataset_name/$model_hyperparameters/$model_state.txt

    #change log from txt to jpg format
    soffice --convert-to jpg ./log/$batik_dataset_name/$experiment_name/$model_state.txt --outdir ./log/$batik_dataset_name/$experiment_name/

    msg="Training ${batik_dataset_name} with ${experiment_name}"

    git add --all
    git commit -m "$msg"
    git push -f

    echo "Finished" $msg
}

#experiment 1
model_hyperparameters_1=kernel=5,zl_dim=20,zg_dim=40,zp_dim=4,ngf=64,ndf=64,batch_size=25
model_state_1=MLP,samefakeimg,G_upsampleConv2d,instance_noise_mean=0.1,shuffle_ds=False,real_label_smoothing=1,nBlocksG=2,nBlocksG_padding=reflect

#experiment 2
model_hyperparameters_2=kernel=5,zl_dim=20,zg_dim=40,zp_dim=4,ngf=64,ndf=64,batch_size=25
model_state_2=MLP,samefakeimg,G_upsampleConv2d,instance_noise_mean=0.1,shuffle_ds=False,real_label_smoothing=1,nBlocksG=2


training 128_Kawung $model_hyperparameters_1 $model_state_1 2
training 128_Parang $model_hyperparameters_1 $model_state_1 2
training 128_Nitik $model_hyperparameters_1 $model_state_1 2
training 128_Lereng $model_hyperparameters_1 $model_state_1 2
training 128_Ceplok $model_hyperparameters_1 $model_state_1 2

sudo poweroff
