export DATE=0830
export SURVEY_NAME="American_Trends_Panel_W$1"
export PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date0828_personas_${SURVEY_NAME}_testonly_haiku.json
export NEW_PERSONA_VAL_FILENAME=opinions_qa/persona_val/${SURVEY_NAME}/date${DATE}_personas_testonly_haiku_bn_hcbds.json
export PREDICT_MODEL="haiku"


# argparser.add_argument("--persona_filename", type=str)
# argparser.add_argument("--new_persona_filename", type=str)
# argparser.add_argument("--method", type=str)
# argparser.add_argument("--score", type=str)
# argparser.add_argument("--em_k", type=int)
# history + demo + persona
for i in 1
do
python -m src.bn_update_persona \
    --persona_filename $PERSONA_VAL_FILENAME \
    --new_persona_filename $NEW_PERSONA_VAL_FILENAME \
    --method hc \
    --score bds \
    --em_k 1
done
