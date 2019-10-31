python3 main.py --mode=train \
    --exp_name=wikigen_demo_2 \
    --type_conditional_lm \
    --max_src_words=40 \
    --debug \
    --task=wikigen \
    --batch_size=20 \
    --num_train_epochs=20 \
    --logging_freq=2
