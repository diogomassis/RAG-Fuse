# overrides
data=dblp_is_lssm
model=RetrieverBERT
text_max_length=256
label_max_length=256
label_enhancement=NONE
text_features_source=TXT

## sparse_retrieve
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py data.num_workers=0 data.batch_size=16 \
    tasks=[sparse_retrieve] \
    model=BM25 \
    data=$data \
    data.text_features_source="$text_features_source" \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/sparse_retrieve_${data}_${fold_idx}.tmr
done

# dense_retrieve fit
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py data.num_workers=0 data.batch_size=16 \
    tasks=[fit] \
    trainer.max_epochs=5 \
    trainer.patience=3 \
    model=$model \
    model.name=LLM_${model} \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=NONE \
    data.text_features_source="$text_features_source" \
    data.batch_size=16 \
    data.num_workers=0 \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/fit_LLM_${model}_${data}_${fold_idx}.tmr
done

# dense_retrieve predict
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py data.num_workers=0 data.batch_size=16 \
    tasks=[predict] \
    trainer.max_epochs=5 \
    trainer.patience=3 \
    model=$model \
    model.name=LLM_${model} \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=NONE \
    data.text_features_source="$text_features_source" \
    data.batch_size=16 \
    data.num_workers=0 \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/predict_LLM_${model}_${data}_${fold_idx}.tmr
done

# dense_retrieve eval
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py data.num_workers=0 data.batch_size=16 \
    tasks=[eval] \
    trainer.max_epochs=5 \
    trainer.patience=3 \
    model=$model \
    model.name=LLM_${model} \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=NONE \
    data.text_features_source="$text_features_source" \
    data.batch_size=16 \
    data.num_workers=0 \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/eval_LLM_${model}_${data}_${fold_idx}.tmr
done

# fuse
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py data.num_workers=0 data.batch_size=16 \
    tasks=[fuse] \
    model=$model \
    model.name=LLM_${model} \
    data=$data \
    data.text_features_source="$text_features_source" \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/fuse_LLM_${model}_${data}_${fold_idx}.tmr
done


# aggregate
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py data.num_workers=0 data.batch_size=16 \
    tasks=[aggregate] \
    model=$model \
    model.name=LLM_${model} \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=NONE \
    data.text_features_source="$text_features_source" \
    data.batch_size=16 \
    data.num_workers=0 \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/aggregate_LLM_${model}_${data}_${fold_idx}.tmr
done