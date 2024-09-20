export DATE=0920
export SURVEY_NAME="American_Trends_Panel_W$1"
export PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date0920_personas_full_personadb.json
export SCORE=$2
export NEW_PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date${DATE}_personas_full_personadb_bn_hc${SCORE}.json
export PREDICT_MODEL="haiku"

for i in 1
do
python -m src.bn_update_persona \
    --persona_filename $PERSONA_VAL_FILENAME \
    --new_persona_filename $NEW_PERSONA_VAL_FILENAME \
    --method hc \
    --score ${SCORE} \
    --em_k 1
done
