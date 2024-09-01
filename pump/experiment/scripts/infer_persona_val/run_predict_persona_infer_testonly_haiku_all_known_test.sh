export SURVEY_NAME="American_Trends_Panel_W$1"

# generate with haiku, full
python -m src.predict \
    --exp_setting persona_infer \
    --survey_name ${SURVEY_NAME} \
    --log_name _unused \
    --persona_path_name date0831_haiku_kmeans20_single_example_known_test \
    --persona_filename opinions_qa/persona_val/${SURVEY_NAME}/date0831_personas_testonly_haiku_known_test.json \
    --persona_inference_model_name haiku
