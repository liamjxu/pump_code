export DATE=0902
export SURVEY_NAME="American_Trends_Panel_W$1"
export PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date0831_personas_full_haiku_known_test.json
export NEW_PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date${DATE}_personas_full_haiku_known_test_bn_hcbds.json
export PREDICT_MODEL="haiku"

for i in 1
do
python -m src.bn_update_persona \
    --persona_filename $PERSONA_VAL_FILENAME \
    --new_persona_filename $NEW_PERSONA_VAL_FILENAME \
    --method hc \
    --score bds \
    --em_k 1
done
