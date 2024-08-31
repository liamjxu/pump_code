export SURVEY_NAME="American_Trends_Panel_W$1"

# generate with haiku, full
python -m src.predict \
    --exp_setting persona_infer_full \
    --survey_name ${SURVEY_NAME} \
    --log_name _unused \
    --persona_filename opinions_qa/persona_val/${SURVEY_NAME}/date0831_personas_full_haiku.json \
    --persona_inference_model_name haiku
