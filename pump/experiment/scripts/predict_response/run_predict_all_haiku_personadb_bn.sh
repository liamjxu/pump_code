export DATE=0911
export SURVEY_NAME="American_Trends_Panel_W$1"
export PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date0904_personas_full_personadb_bn_hcbds.json
export PREDICT_MODEL=haiku
if [ "$1" -eq 34 ]; then
    # export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0905_personas_full_personadb_querydp_trainAll_top80_skew10_withname.json"
    export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0905_personas_full_personadb_bn_hcbdeu_querydp_trainAll_top80_skew10_withname.json"
elif [ "$1" -eq 41 ]; then
    # export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0905_personas_full_personadb_bn_queryponly_trainAll_top80_skew3_withname.json"
    export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0905_personas_full_personadb_bn_queryponly_trainAll_top160_skew15_withname.json"
elif [ "$1" -eq 82 ]; then
    # export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0905_personas_full_personadb_querydp_trainAll_top80_skew7_withname.json"
    # below: tried, bad
    # export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0905_personas_full_personadb_bn_querydp_trainAll_top140_skew2_withname.json"
    # export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0905_personas_full_personadb_bn_querydp_trainAll_top100_skew2_withname.json"  # 580 bdeu
    export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0905_personas_full_personadb_bn_querydp_trainAll_top80_skew2_withname.json"  # 585 bic
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

# # history + demo + persona
# python -m src.predict \
#     --using_personadb_surveys \
#     --exp_setting history_demo_persona \
#     --survey_name ${SURVEY_NAME} \
#     --persona_level low mid high \
#     --response_prediction_model_name ${PREDICT_MODEL} \
#     --persona_repr namedescvalue \
#     --log_name ${SURVEY_NAME}/date${DATE}_v17_personadb_surveys_bdemo_bn_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_history_demo_persona_run${i}.json \
#     --persona_filename ${PERSONA_VAL_FILENAME} 
done

# # history + demo
# for i in 1 2 3
# do
# python -m src.predict \
#     --using_personadb_surveys \
#     --exp_setting history_demo \
#     --survey_name ${SURVEY_NAME} \
#     --persona_level low mid high \
#     --response_prediction_model_name ${PREDICT_MODEL} \
#     --persona_repr namedescvalue \
#     --log_name ${SURVEY_NAME}/date${DATE}_v17_personadb_surveys_bdemo_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_history_demo_run${i}.json \
#     --persona_filename ${PERSONA_VAL_FILENAME} 
# done

# # history
# for i in 1 2 3
# do
# python -m src.predict \
#     --using_personadb_surveys \
#     --exp_setting history \
#     --survey_name ${SURVEY_NAME} \
#     --persona_level low mid high \
#     --response_prediction_model_name ${PREDICT_MODEL} \
#     --persona_repr namedescvalue \
#     --log_name ${SURVEY_NAME}/date${DATE}_v17_personadb_surveys_bdemo_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_history_run${i}.json \
#     --persona_filename ${PERSONA_VAL_FILENAME} 
# done

# # history + persona
# for i in 1 2 3
# do
# python -m src.predict \
#     --using_personadb_surveys \
#     --exp_setting history_persona \
#     --survey_name ${SURVEY_NAME} \
#     --persona_level low mid high \
#     --response_prediction_model_name ${PREDICT_MODEL} \
#     --persona_repr namedescvalue \
#     --log_name ${SURVEY_NAME}/date${DATE}_v17_personadb_surveys_bdemo_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_history_persona_run${i}.json \
#     --persona_filename ${PERSONA_VAL_FILENAME} 
# done

# # demo
# for i in 1 2 3
# do
# python -m src.predict \
#     --using_personadb_surveys \
#     --exp_setting demo \
#     --survey_name ${SURVEY_NAME} \
#     --persona_level low mid high \
#     --response_prediction_model_name ${PREDICT_MODEL} \
#     --persona_repr namedescvalue \
#     --log_name ${SURVEY_NAME}/date${DATE}_v17_personadb_surveys_bdemo_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_demo_run${i}.json \
#     --persona_filename ${PERSONA_VAL_FILENAME} 
# done

# # persona
# for i in 1 2 3
# do
# python -m src.predict \
#     --using_personadb_surveys \
#     --exp_setting persona \
#     --survey_name ${SURVEY_NAME} \
#     --persona_level low mid high \
#     --response_prediction_model_name ${PREDICT_MODEL} \
#     --persona_repr namedescvalue \
#     --log_name ${SURVEY_NAME}/date${DATE}_v17_personadb_surveys_bdemo_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_persona_run${i}.json \
#     --persona_filename ${PERSONA_VAL_FILENAME} 
# done

# # demo + persona
# for i in 1 2 3
# do
# python -m src.predict \
#     --using_personadb_surveys \
#     --exp_setting demo_persona \
#     --survey_name ${SURVEY_NAME} \
#     --persona_level low mid high \
#     --response_prediction_model_name ${PREDICT_MODEL} \
#     --persona_repr namedescvalue \
#     --log_name ${SURVEY_NAME}/date${DATE}_v17_personadb_surveys_bdemo_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_demo_persona_run${i}.json \
#     --persona_filename ${PERSONA_VAL_FILENAME} 
# done
