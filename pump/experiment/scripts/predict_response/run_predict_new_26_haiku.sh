export DATE=0828
export SURVEY_NAME="American_Trends_Panel_W26"
export PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date${DATE}_personas_${SURVEY_NAME}_testonly_haiku.json


# history + demo + persona
for i in 1
do
python -m src.predict \
    --exp_setting history_demo_persona \
    --survey_name ${SURVEY_NAME} \
    --persona_level low mid high \
    --persona_repr namedescvalue \
    --log_name ${SURVEY_NAME}/date${DATE}_v10post_midterm_${SURVEY_NAME}_haikupred_history_demo_persona_run${i}.json \
    --persona_filename ${PERSONA_VAL_FILENAME} 
done

# history
for i in 1
do
python -m src.predict \
    --exp_setting history \
    --survey_name ${SURVEY_NAME} \
    --persona_level low mid high \
    --persona_repr namedescvalue \
    --log_name ${SURVEY_NAME}/date${DATE}_v10post_midterm_${SURVEY_NAME}_haikupred_history_run${i}.json \
    --persona_filename ${PERSONA_VAL_FILENAME} 
done

# demo
for i in 1
do
python -m src.predict \
    --exp_setting demo \
    --survey_name ${SURVEY_NAME} \
    --persona_level low mid high \
    --persona_repr namedescvalue \
    --log_name ${SURVEY_NAME}/date${DATE}_v10post_midterm_${SURVEY_NAME}_haikupred_demo_run${i}.json \
    --persona_filename ${PERSONA_VAL_FILENAME} 
done

# history + demo
for i in 1
do
python -m src.predict \
    --exp_setting history_demo \
    --survey_name ${SURVEY_NAME} \
    --persona_level low mid high \
    --persona_repr namedescvalue \
    --log_name ${SURVEY_NAME}/date${DATE}_v10post_midterm_${SURVEY_NAME}_haikupred_history_demo_run${i}.json \
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
    --log_name ${SURVEY_NAME}/date${DATE}_v10post_midterm_${SURVEY_NAME}_haikupred_persona_run${i}.json \
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
    --log_name ${SURVEY_NAME}/date${DATE}_v10post_midterm_${SURVEY_NAME}_haikupred_history_persona_run${i}.json \
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
    --log_name ${SURVEY_NAME}/date${DATE}_v10post_midterm_${SURVEY_NAME}_haikupred_demo_persona_run${i}.json \
    --persona_filename ${PERSONA_VAL_FILENAME} 
done
