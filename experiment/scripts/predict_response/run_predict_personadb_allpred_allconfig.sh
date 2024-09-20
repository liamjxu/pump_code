export DATE=0920
export SURVEY_NAME="American_Trends_Panel_W$1"
export PREDICT_MODEL=$2
if [ "$1" -eq 34 ]; then
    export PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date0920_personas_full_personadb_bn_hcbds.json
    export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0920_personas_full_personadb_bn_hcbds_querydp_trainAll_top80_skew10_withname.json"
elif [ "$1" -eq 41 ]; then
    export PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date0920_personas_full_personadb_bn_hcbic.json
    export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0920_personas_full_personadb_bn_hcbic_queryponly_trainAll_top160_skew15_withname.json"
elif [ "$1" -eq 82 ]; then
    export PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date0920_personas_full_personadb_bn_hcbdeu.json
    export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0920_personas_full_personadb_bn_hcbdeu_querydp_trainAll_top100_skew2_withname.json" 
else
    echo "Invalid input. Please provide 34 or 41 or 82 as the argument."
fi

for i in 1 2 3
do
# history + demo + persona + rag
python -m src.predict \
    --using_personadb_surveys \
    --exp_setting history_demo_persona_rag \
    --survey_name ${SURVEY_NAME} \
    --persona_level low mid high \
    --response_prediction_model_name ${PREDICT_MODEL} \
    --rag_similar_user_mapping ${SIMILAR_USER_MAPPING_FILENAME} \
    --persona_repr namedescvalue \
    --log_name ${SURVEY_NAME}/date${DATE}_v17_personadb_surveys_bdemo_mostcommon40_bn_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_history_demo_persona_rag_run${i}.json \
    --persona_filename ${PERSONA_VAL_FILENAME} 

# history
python -m src.predict \
    --using_personadb_surveys \
    --exp_setting history \
    --survey_name ${SURVEY_NAME} \
    --persona_level low mid high \
    --response_prediction_model_name ${PREDICT_MODEL} \
    --persona_repr namedescvalue \
    --log_name ${SURVEY_NAME}/date${DATE}_v17_personadb_surveys_bdemo_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_history_run${i}.json \
    --persona_filename ${PERSONA_VAL_FILENAME} 

# history + demo + persona
python -m src.predict \
    --using_personadb_surveys \
    --exp_setting history_demo_persona \
    --survey_name ${SURVEY_NAME} \
    --persona_level low mid high \
    --response_prediction_model_name ${PREDICT_MODEL} \
    --persona_repr namedescvalue \
    --log_name ${SURVEY_NAME}/date${DATE}_v17_personadb_surveys_bdemo_bn_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_history_demo_persona_run${i}.json \
    --persona_filename ${PERSONA_VAL_FILENAME} 

# history + demo
python -m src.predict \
    --using_personadb_surveys \
    --exp_setting history_demo \
    --survey_name ${SURVEY_NAME} \
    --persona_level low mid high \
    --response_prediction_model_name ${PREDICT_MODEL} \
    --persona_repr namedescvalue \
    --log_name ${SURVEY_NAME}/date${DATE}_v17_personadb_surveys_bdemo_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_history_demo_run${i}.json \
    --persona_filename ${PERSONA_VAL_FILENAME}
done
