# # generate with haiku
# python -m experiment.predict \
#     --exp_setting persona_infer_full \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name _unused \
#     --persona_filename date0729_midterm_personas_full_haiku.json \
#     --persona_inference_model_name haiku

# # generate with sonnet
# python -m experiment.predict \
#     --exp_setting persona_infer_full \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name _unused \
#     --persona_filename date0729_midterm_personas_full_sonnet.json \
#     --persona_inference_model_name sonnet


# # generate with haiku, more surveys
# python -m src.predict \
#     --exp_setting persona_infer_full \
#     --survey_name American_Trends_Panel_W27 \
#     --log_name _unused \
#     --persona_filename opinions_qa/persona_val/date0825_personas_full_haiku_American_Trends_Panel_W27.json \
#     --persona_inference_model_name haiku

# python -m src.predict \
#     --exp_setting persona_infer_full \
#     --survey_name American_Trends_Panel_W32 \
#     --log_name _unused \
#     --persona_filename opinions_qa/persona_val/date0825_personas_full_haiku_American_Trends_Panel_W32.json \
#     --persona_inference_model_name haiku
