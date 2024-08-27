export DATE=0826
export SURVEY_NAME="American_Trends_Panel_W26"
export PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date0826_personas_full_sonnet_bn_em.json


# history + demo + persona
for i in 1
do
python -m src.predict \
    --exp_setting history_demo_persona \
    --survey_name ${SURVEY_NAME} \
    --persona_level low mid high \
    --persona_repr namedescvalue \
    --log_name ${SURVEY_NAME}/date${DATE}_v12_bnem_${SURVEY_NAME}_history_demo_persona_run${i}.json \
    --persona_filename ${PERSONA_VAL_FILENAME} 
done

# history + persona
for i in 1
do
python -m src.predict \
    --exp_setting history_persona \
    --survey_name ${SURVEY_NAME} \
    --persona_level low mid high \
    --persona_repr namedescvalue \
    --log_name ${SURVEY_NAME}/date${DATE}_v12_bnem_${SURVEY_NAME}_history_persona_run${i}.json \
    --persona_filename ${PERSONA_VAL_FILENAME} 
done

# persona
for i in 1
do
python -m src.predict \
    --exp_setting persona \
    --survey_name ${SURVEY_NAME} \
    --persona_level low mid high \
    --persona_repr namedescvalue \
    --log_name ${SURVEY_NAME}/date${DATE}_v12_bnem_${SURVEY_NAME}_persona_run${i}.json \
    --persona_filename ${PERSONA_VAL_FILENAME} 
done

# demo + persona
for i in 1
do
python -m src.predict \
    --exp_setting demo_persona \
    --survey_name ${SURVEY_NAME} \
    --persona_level low mid high \
    --persona_repr namedescvalue \
    --log_name ${SURVEY_NAME}/date${DATE}_v12_bnem_${SURVEY_NAME}_demo_persona_run${i}.json \
    --persona_filename ${PERSONA_VAL_FILENAME} 
done