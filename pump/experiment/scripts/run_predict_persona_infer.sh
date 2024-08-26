# generate with haiku
python -m experiment.predict \
    --exp_setting persona_infer_full \
    --survey_name American_Trends_Panel_W26 \
    --log_name _unused \
    --persona_filename date0729_midterm_personas_full_haiku.json \
    --persona_inference_model_name haiku

# generate with sonnet
python -m experiment.predict \
    --exp_setting persona_infer_full \
    --survey_name American_Trends_Panel_W26 \
    --log_name _unused \
    --persona_filename date0729_midterm_personas_full_sonnet.json \
    --persona_inference_model_name sonnet