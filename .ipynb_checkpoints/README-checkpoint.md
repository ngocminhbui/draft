# draft

## ring classifier:
train file: best_shrec_models/shrec2017/ring_classifier/square_ring_classifier_v1_train_full.py


python square_ring_classifier_v1_train_full.py --config_name <your_config> --data <data_folder_path> --log_folder <log_folder_path> --learning_rate 0.00001 


Best weight (model) trained: model_best_testset.ckpt-118420.*
(resume,load, default: none: --resume model_best_testset.ckpt-118420 )

## ring score:
train file: best_shrec_models/shrec2017/score_net/exec_file.py (square_ring_setting)


python exec_file.py --config_name <your_config> --data <data_folder_path> --log_folder <log_folder_path> --learning_rate 0.00001 


Best weight (model) trained: model.ckpt-40273.*
(resume,load, default: none: --resume model.ckpt-40273 )

## fusion ring result & ring score:
code file: best_shrec_models/shrec2017/ring_classifier/combine_attention_and_class_score.ipynb

require inputs input from ring score & ring classifier:
+ ring classify score: 20 class scores for each rings. from ring classifier. ex:./shr17/s_v1/attention_eval.score.40273.0.txt
+ ring attention score: 1 scalar score for each ring. from ring score. ex: ./shr17/s_v1_fullt_train_val/model_best_testset_118420.0.score.txt
