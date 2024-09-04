export TRAIN_SIZE=2000
export TOP_K=40
export SKEW=10
export USEDEMO=True

python -m src.embed_persona \
    --survey_name American_Trends_Panel_W34 \
    --persona_val_path opinions_qa/persona_val/American_Trends_Panel_W34/date0902_personas_full_haiku_known_test_bn_hcbds.json \
    --user_mapping_filepath opinions_qa/similar_users/American_Trends_Panel_W34/date0902_personas_full_haiku_known_test_bn_hcbds_train${TRAIN_SIZE}_top${TOP_K}_skew${SKEW}_usedemo${USEDEMO}_withname.json \
    --train_size $TRAIN_SIZE \
    --top_k $TOP_K \
    --skew $SKEW \
    --use_demo



# export TRAIN_SIZE=200
# export TOP_K=20

# python -m src.embed_persona \
#     --survey_name American_Trends_Panel_W34 \
#     --persona_val_path opinions_qa/persona_val/American_Trends_Panel_W34/date0901_personas_testonly_haiku_known_test_use_all_q.json \
#     --user_mapping_filepath opinions_qa/similar_users/American_Trends_Panel_W34/date0831_personas_full_haiku_known_test_train${TRAIN_SIZE}_top${TOP_K}_withname_allq.json \
#     --train_size $TRAIN_SIZE \
#     --top_k $TOP_K