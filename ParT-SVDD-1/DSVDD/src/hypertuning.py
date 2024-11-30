#!/usr/bin/env python3


import optuna
import subprocess
import json
import glob
import os
import shutil
import matplotlib.pyplot as plt


from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_rank
from optuna.visualization.matplotlib import plot_slice
from optuna.visualization.matplotlib import plot_timeline

#moving the data into a backup folder before the code below is run, using glob or os, maybe walk, scan directiory

base_dir ="/data/alice/tqwsavelkoel/ParT-SVDD-1/DSVDD/src/training/Test/"

backup_dir = os.path.join(base_dir,'backup')

def movefiles_backup():

    os.makedirs(backup_dir, exist_ok=True)

    for folder in glob.glob(os.path.join(base_dir, "*/")):
        name_folder = os.path.basename(os.path.normpath(folder))
        if name_folder =="backup":
            continue

        day = name_folder
        runs_on_day=glob.glob(os.path.join(backup_dir, f"{day}-*"))
        backup_number = len(runs_on_day) + 1
        backup_name = f"{day}-{backup_number}"

        folder_path =os.path.join(backup_dir,backup_name)
        shutil.move(folder,folder_path)

def save_study_plots(study, study_folder):
    plots_dir = os.path.join(study_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_optimization_history(study).figure.savefig(os.path.join(plots_dir, "optimization_history.png"),bbox_inches='tight')
    plot_parallel_coordinate(study).figure.savefig(os.path.join(plots_dir, "parallel_coordinate.png"),bbox_inches='tight')
    #plot_slice(study).figure.savefig(os.path.join(plots_dir, "slice.png"))


    params = ["lr", "weight_decay", "nu", "delta", "epsilon"]
    for para in params:
        
        slice_plot =optuna.visualization.matplotlib.plot_slice(study, params=[f"{para}"])
        slice_plot.figure.savefig(os.path.join(plots_dir, f"slice_{para}.png"),bbox_inches='tight')


    optuna.visualization.matplotlib.plot_param_importances(study).figure.savefig(os.path.join(plots_dir, "param_importances.png"),bbox_inches='tight')

    #optuna.visualization.matplotlib.plot_param_importances(study).figure.savefig(os.path.join(plots_dir, "param_importances.png"),bbox_inches='tight')

    #slice_plot =optuna.visualization.matplotlib.plot_slice(study)
    #slice_plot[0].figure.savefig(os.path.join(plots_dir, "slice.png"),bbox_inches='tight')

    


def objective(trial):
    #the hyperparmeters that need tuning
    lr =trial.suggest_float("lr", 1e-7, 1e-1, log=True)
    weight_decay =trial.suggest_float("weight_decay", 1e-7, 1e-1, log=True)
    nu =trial.suggest_float("nu", 0.01, 1)
    delta = trial.suggest_float("delta", 1e-7, 1e-1, log=True)
    epsilon =trial.suggest_float("epsilon", 1e-10, 1e-1, log=True)

    task = ["bash",'ParT-SVDD.sh',str(lr),str(weight_decay),str(nu),str(delta),str(epsilon)]

    subprocess.run(task, capture_output=True, text=True)

    recent_run_day = max(glob.glob(os.path.join(base_dir,"*/")),key=os.path.getmtime)
    recent_run_time = max(glob.glob(os.path.join(recent_run_day,"*/")),key=os.path.getmtime)


    results_json_path = os.path.join(recent_run_time,"predict_output/results.json")

    with open(results_json_path,"r") as fp:
        results = json.load(fp) 
        loss_list= results["loss"]
        loss = loss_list[-1]

    return loss


movefiles_backup()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=2)

recent_study_day = max(glob.glob(os.path.join(base_dir, "*/")), key=os.path.getmtime)
recent_study_folder = os.path.join(recent_study_day)

save_study_plots(study, recent_study_folder)

print(study.best_params)


 

def simplerun():
    lr = 0.0001
    weight_decay = 0.5e-6
    nu = 0.02
    delta = 1e-5
    epsilon = 1e-8

    task = ["bash","ParT-SVDD.sh", str(lr), str(weight_decay), str(nu), str(delta), str(epsilon)]
    print("Running the following task:", task)

    result =subprocess.run(task, capture_output=True, text=True)
    #print("STDOUT:", result.stdout)
    #print("STDERR:", result.stderr)

#simplerun()
#plot_optimization_history(study)
