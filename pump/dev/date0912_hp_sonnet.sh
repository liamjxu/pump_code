export DATE=0912
export SURVEY_NAME="American_Trends_Panel_W$1"
export PREDICT_MODEL=sonnet
export BN=$2
export CONFIG=$3

if [ "$1" -eq 34 ]; then
    if [ "$2" = "bds" ]; then
        export PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date0904_personas_full_personadb_bn_hcbds.json
        if [ "$3" -eq 1 ]; then
            export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0912_personas_full_personadb_bn_hcbds_querydp_trainAll_top100_skew10_withname.json"
        elif [ "$3" -eq 2 ]; then
            export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0912_personas_full_personadb_bn_hcbds_querydp_trainAll_top80_skew5_withname.json"
        fi
    elif [ "$2" = "bdeu" ]; then
        export PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date0904_personas_full_personadb_bn_hcbdeu.json
        if [ "$3" -eq 1 ]; then
            export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0912_personas_full_personadb_bn_hcbdeu_querydp_trainAll_top80_skew10_withname.json"
        elif [ "$3" -eq 2 ]; then
            export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0912_personas_full_personadb_bn_hcbdeu_querydp_trainAll_top80_skew7_withname.json"
        elif [ "$3" -eq 3 ]; then
            export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0912_personas_full_personadb_bn_hcbdeu_idx132_querydp_trainAll_top60_skew3_withname.json"
        elif [ "$3" -eq 4 ]; then
            export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0912_personas_full_personadb_bn_hcbdeu_idx108_querydp_trainAll_top100_skew10_withname.json"
        fi
    elif [ "$2" = "bic" ]; then
        export PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date0904_personas_full_personadb_bn_hcbic.json
        if [ "$3" -eq 1 ]; then
            export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0912_personas_full_personadb_bn_hcbic_querydp_trainAll_top100_skew2_withname.json"
        elif [ "$3" -eq 2 ]; then
            export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0912_personas_full_personadb_bn_hcbic_querydp_trainAll_top100_skew10_withname.json"
        fi
    else
        export PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date0904_personas_full_personadb.json
        if [ "$3" -eq 1 ]; then
            export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0912_personas_full_personadb_idx2_querydp_trainAll_top80_skew10_withname.json"
        elif [ "$3" -eq 2 ]; then
            export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0912_personas_full_personadb_idx1_querydp_trainAll_top60_skew10_withname.json"
        fi

    fi
# elif [ "$1" -eq 41 ]; then
#     # 56.1 / 55.5
#     export PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date0904_personas_full_personadb_bn_hcbic.json
#     export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0905_personas_full_personadb_bn_queryponly_trainAll_top160_skew15_withname.json"
# elif [ "$1" -eq 82 ]; then
#     # 58.3 / 58.3
#     export PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date0904_personas_full_personadb_bn_hcbdeu.json
#     export SIMILAR_USER_MAPPING_FILENAME="opinions_qa/similar_users/${SURVEY_NAME}/date0905_personas_full_personadb_bn_hcbdeu_querydp_trainAll_top100_skew2_withname.json" 
else
    echo "Invalid input. Please provide 34 or 41 or 82 as the argument."
fi

for i in 1 2
do
# # history + demo + persona + rag
# python -m src.predict \
#     --using_personadb_surveys \
#     --exp_setting history_demo_persona_rag \
#     --survey_name ${SURVEY_NAME} \
#     --persona_level low mid high \
#     --response_prediction_model_name ${PREDICT_MODEL} \
#     --rag_similar_user_mapping ${SIMILAR_USER_MAPPING_FILENAME} \
#     --persona_repr namedescvalue \
#     --log_name ${SURVEY_NAME}/date${DATE}_v17_personadb_surveys_bdemo_mostcommon40_bn${BN}${CONFIG}_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_history_demo_persona_rag_run${i}.json \
#     --persona_filename ${PERSONA_VAL_FILENAME} 

# history + demo + rag
python -m src.predict \
    --using_personadb_surveys \
    --exp_setting history_demo_rag \
    --survey_name ${SURVEY_NAME} \
    --persona_level low mid high \
    --response_prediction_model_name ${PREDICT_MODEL} \
    --rag_similar_user_mapping ${SIMILAR_USER_MAPPING_FILENAME} \
    --persona_repr namedescvalue \
    --log_name ${SURVEY_NAME}/date${DATE}_v17_personadb_surveys_bdemo_mostcommon40_bn${BN}${CONFIG}_${SURVEY_NAME}_${PREDICT_MODEL}pred_prompt3_history_demo_rag_run${i}.json \
    --persona_filename ${PERSONA_VAL_FILENAME}

done
