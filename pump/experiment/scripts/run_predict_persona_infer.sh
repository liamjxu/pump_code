# python -m experiment.predict \
#     --exp_setting persona_infer \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name unused \
#     --persona_filename date0729_midterm_personas.json 

# python -m experiment.predict \
#     --exp_setting persona_infer \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name unused \
#     --persona_filename date0729_midterm_personas_haiku.json \
#     --persona_inference_model_name haiku

# python -m experiment.predict \
#     --exp_setting persona_infer_full \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name unused \
#     --persona_filename date0729_midterm_personas_full_haiku.json \
#     --persona_inference_model_name haiku

python -m experiment.predict \
    --exp_setting persona_infer_full \
    --survey_name American_Trends_Panel_W26 \
    --log_name unused \
    --persona_filename date0729_midterm_personas_full_sonnet.json \
    --persona_inference_model_name sonnet