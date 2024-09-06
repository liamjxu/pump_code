export SURVEY_NAME=American_Trends_Panel_W$1
export PERSONA_VAL_PATH=opinions_qa/persona_val/$SURVEY_NAME/date0904_personas_full_personadb.json

python -m dev.hp_grid_search \
    --survey_name $SURVEY_NAME \
    --persona_val_path $PERSONA_VAL_PATH \
    --using_personadb_surveys

