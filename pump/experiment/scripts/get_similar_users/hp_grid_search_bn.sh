export DATE=0904
export SURVEY_NAME=American_Trends_Panel_W$1
export SCORE=$2
export PERSONA_VAL_PATH=opinions_qa/persona_val/$SURVEY_NAME/date${DATE}_personas_full_personadb_bn_hc${SCORE}.json
export OUTPUT_NAME=opinions_qa/similar_users/$SURVEY_NAME/date${DATE}_personas_full_personadb_bn_hc${SCORE}_grid_search.json

python -m src.hp_grid_search \
    --using_personadb_surveys \
    --survey_name $SURVEY_NAME \
    --persona_val_path $PERSONA_VAL_PATH \
    --output_name $OUTPUT_NAME


