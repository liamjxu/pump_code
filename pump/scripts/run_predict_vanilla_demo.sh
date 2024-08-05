# python -m experiment.predict \
#     --exp_setting vanilla_demo \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name date0729_vanilla_demo_American_Trends_Panel_W26.json


############### 0802

python -m experiment.predict \
    --use_only_relevant_persona \
    --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
    --exp_setting vanilla_demo \
    --survey_name American_Trends_Panel_W26 \
    --log_name date0802_vanilla_demo_v6all_namevalue_selected_sonnetinfer_American_Trends_Panel_W26.json \
    --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json
