# draft

# ring classifier:
train file: best_shrec_models/shrec2017/ring_classifier/square_ring_classifier_v1_train_full.py


python square_ring_classifier_v1_train_full.py --config_name <your_config> --data <data_folder_path> --log_folder <log_folder_path> --learning_rate 0.00001 


Best weight (model) trained: model_best_testset.ckpt-118420.*
(resume,load, default: none: --resume model_best_testset.ckpt-118420 )

# ring score:
train file: best_shrec_models/shrec2017/score_net/exec_file.py (square_ring_setting)


python exec_file.py --config_name <your_config> --data <data_folder_path> --log_folder <log_folder_path> --learning_rate 0.00001 


Best weight (model) trained: model.ckpt-40273.*
(resume,load, default: none: --resume model.ckpt-40273 )




