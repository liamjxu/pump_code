export DATE=0901
export SURVEY_NAME="American_Trends_Panel_W$1"
export PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date0901_personas_testonly_haiku_known_test_use_all_q.json
export TRAIN_SIZE=200
export TOP_K=20
export SIMILAR_USER_MAPPING_FILENAME=opinions_qa/similar_users/${SURVEY_NAME}/date0831_personas_full_haiku_known_test_train${TRAIN_SIZE}_top${TOP_K}_withname_allq.json
export PREDICT_MODEL="haiku"


# history + demo + persona
for i in 1
do
python -m src.predict \
    --exp_setting history_demo_persona \
    --survey_name ${SURVEY_NAME} \
    --persona_level low mid high \
    --response_prediction_model_name ${PREDICT_MODEL} \
    --rag_similar_user_mapping ${SIMILAR_USER_MAPPING_FILENAME} \
    --persona_repr namedescvalue \
    --log_name ${SURVEY_NAME}/date${DATE}_v13_known_test_allq_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_history_demo_persona_run${i}.json \
    --persona_filename ${PERSONA_VAL_FILENAME} 
done

# # history + demo + persona + RAG
# for i in 1
# do
# python -m src.predict \
#     --exp_setting history_demo_persona_rag \
#     --survey_name ${SURVEY_NAME} \
#     --persona_level low mid high \
#     --response_prediction_model_name ${PREDICT_MODEL} \
#     --rag_similar_user_mapping ${SIMILAR_USER_MAPPING_FILENAME} \
#     --persona_repr namedescvalue \
#     --log_name ${SURVEY_NAME}/date${DATE}_v16_known_test_rag_${TRAIN_SIZE}_${TOP_K}_withname_allq_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_history_demo_persona_run${i}.json \
#     --persona_filename ${PERSONA_VAL_FILENAME} 
# done

# # history
# for i in 1
# do
# python -m src.predict \
#     --exp_setting history \
#     --survey_name ${SURVEY_NAME} \
#     --persona_level low mid high \
#     --response_prediction_model_name ${PREDICT_MODEL} \
#     --persona_repr namedescvalue \
#     --log_name ${SURVEY_NAME}/date${DATE}_v16_known_test_rag_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_history_run${i}.json \
#     --persona_filename ${PERSONA_VAL_FILENAME} 
# done

# # history + persona
# for i in 1
# do
# python -m src.predict \
#     --exp_setting history_persona \
#     --survey_name ${SURVEY_NAME} \
#     --persona_level low mid high \
#     --response_prediction_model_name ${PREDICT_MODEL} \
#     --persona_repr namedescvalue \
#     --log_name ${SURVEY_NAME}/date${DATE}_v16_known_test_rag_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_history_persona_run${i}.json \
#     --persona_filename ${PERSONA_VAL_FILENAME} 
# done

# # demo
# for i in 1
# do
# python -m src.predict \
#     --exp_setting demo \
#     --survey_name ${SURVEY_NAME} \
#     --persona_level low mid high \
#     --response_prediction_model_name ${PREDICT_MODEL} \
#     --persona_repr namedescvalue \
#     --log_name ${SURVEY_NAME}/date${DATE}_v16_known_test_rag_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_demo_run${i}.json \
#     --persona_filename ${PERSONA_VAL_FILENAME} 
# done

# # history + demo
# for i in 1
# do
# python -m src.predict \
#     --exp_setting history_demo \
#     --survey_name ${SURVEY_NAME} \
#     --persona_level low mid high \
#     --response_prediction_model_name ${PREDICT_MODEL} \
#     --persona_repr namedescvalue \
#     --log_name ${SURVEY_NAME}/date${DATE}_v16_known_test_rag_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_history_demo_run${i}.json \
#     --persona_filename ${PERSONA_VAL_FILENAME} 
# done

# # persona
# for i in 1
# do
# python -m src.predict \
#     --exp_setting persona \
#     --survey_name ${SURVEY_NAME} \
#     --persona_level low mid high \
#     --response_prediction_model_name ${PREDICT_MODEL} \
#     --persona_repr namedescvalue \
#     --log_name ${SURVEY_NAME}/date${DATE}_v16_known_test_rag_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_persona_run${i}.json \
#     --persona_filename ${PERSONA_VAL_FILENAME} 
# done

# # demo + persona
# for i in 1
# do
# python -m src.predict \
#     --exp_setting demo_persona \
#     --survey_name ${SURVEY_NAME} \
#     --persona_level low mid high \
#     --response_prediction_model_name ${PREDICT_MODEL} \
#     --persona_repr namedescvalue \
#     --log_name ${SURVEY_NAME}/date${DATE}_v16_known_test_rag_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_demo_persona_run${i}.json \
#     --persona_filename ${PERSONA_VAL_FILENAME} 
# done
