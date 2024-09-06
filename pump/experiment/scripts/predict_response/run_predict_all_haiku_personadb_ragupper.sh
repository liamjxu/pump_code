export DATE=0904
export SURVEY_NAME="American_Trends_Panel_W$1"
export PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date0904_personas_full_personadb.json
export PREDICT_MODEL="haiku"


# history + demo + persona
for i in 1 2 3
do
python -m src.predict \
    --using_personadb_surveys \
    --ragupper \
    --exp_setting history_demo_persona_rag \
    --survey_name ${SURVEY_NAME} \
    --persona_level low mid high \
    --response_prediction_model_name ${PREDICT_MODEL} \
    --persona_repr namedescvalue \
    --log_name ${SURVEY_NAME}/date${DATE}_v17_personadb_surveys_ragupper_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_history_demo_persona_run${i}.json \
    --persona_filename ${PERSONA_VAL_FILENAME} 
done

# history
for i in 1 2 3
do
python -m src.predict \
    --using_personadb_surveys \
    --ragupper \
    --exp_setting history_rag \
    --survey_name ${SURVEY_NAME} \
    --persona_level low mid high \
    --response_prediction_model_name ${PREDICT_MODEL} \
    --persona_repr namedescvalue \
    --log_name ${SURVEY_NAME}/date${DATE}_v17_ragupper_personadb_surveys_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_history_run${i}.json \
    --persona_filename ${PERSONA_VAL_FILENAME} 
done

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
#     --log_name ${SURVEY_NAME}/date${DATE}_v17_personadb_surveys_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_history_persona_run${i}.json \
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
#     --log_name ${SURVEY_NAME}/date${DATE}_v17_personadb_surveys_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_demo_run${i}.json \
#     --persona_filename ${PERSONA_VAL_FILENAME} 
# done

# history + demo
for i in 1 2 3
do
python -m src.predict \
    --using_personadb_surveys \
    --ragupper \
    --exp_setting history_demo_rag \
    --survey_name ${SURVEY_NAME} \
    --persona_level low mid high \
    --response_prediction_model_name ${PREDICT_MODEL} \
    --persona_repr namedescvalue \
    --log_name ${SURVEY_NAME}/date${DATE}_v17_ragupper_personadb_surveys_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_history_demo_run${i}.json \
    --persona_filename ${PERSONA_VAL_FILENAME} 
done

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
#     --log_name ${SURVEY_NAME}/date${DATE}_v17_personadb_surveys_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_persona_run${i}.json \
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
#     --log_name ${SURVEY_NAME}/date${DATE}_v17_personadb_surveys_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_demo_persona_run${i}.json \
#     --persona_filename ${PERSONA_VAL_FILENAME} 
# done
