# # generate with haiku, test only
# python -m src.predict \
#     --exp_setting persona_infer \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name _unused \
#     --persona_filename opinions_qa/persona_val/American_Trends_Panel_W26/date0828_personas_American_Trends_Panel_W26_testonly_haiku.json \
#     --persona_inference_model_name haiku

# python -m src.predict \
#     --exp_setting persona_infer \
#     --survey_name American_Trends_Panel_W27 \
#     --log_name _unused \
#     --persona_filename opinions_qa/persona_val/American_Trends_Panel_W27/date0828_personas_American_Trends_Panel_W27_testonly_haiku.json \
#     --persona_inference_model_name haiku

# python -m src.predict \
#     --exp_setting persona_infer \
#     --survey_name American_Trends_Panel_W29 \
#     --log_name _unused \
#     --persona_filename opinions_qa/persona_val/American_Trends_Panel_W29/date0828_personas_American_Trends_Panel_W29_testonly_haiku.json \
#     --persona_inference_model_name haiku

python -m src.predict \
    --exp_setting persona_infer \
    --survey_name American_Trends_Panel_W32 \
    --log_name _unused \
    --persona_filename opinions_qa/persona_val/American_Trends_Panel_W32/date0828_personas_American_Trends_Panel_W32_testonly_haiku.json \
    --persona_inference_model_name haiku

# python -m src.predict \
#     --exp_setting persona_infer \
#     --survey_name American_Trends_Panel_W34 \
#     --log_name _unused \
#     --persona_filename opinions_qa/persona_val/American_Trends_Panel_W34/date0828_personas_American_Trends_Panel_W34_testonly_haiku.json \
#     --persona_inference_model_name haiku
