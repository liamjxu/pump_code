export SURVEY_NAME="American_Trends_Panel_W$1"

# generate with haiku
python -m src.predict \
    --using_personadb_surveys \
    --exp_setting persona_infer_full \
    --survey_name ${SURVEY_NAME} \
    --log_name _unused \
    --persona_path_name date0904_haiku_kmeans10_single_example_known_test_use_all_q \
    --persona_filename opinions_qa/persona_val/${SURVEY_NAME}/date0904_personas_full_personadb.json \
    --persona_inference_model_name haiku
