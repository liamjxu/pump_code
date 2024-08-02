# python -m experiment.predict \
#     --exp_setting vanilla_no_history_persona \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name date0729_vanilla_no_history_persona_v3low_run2_American_Trends_Panel_W26.json \
#     --persona_filename opinions_qa/persona/date0729_midterm_personas.json

# python -m experiment.predict \
#     --exp_setting vanilla_no_history_persona \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name date0730_vanilla_no_history_persona_v4low_full_haikuinfer_American_Trends_Panel_W26.json \
#     --persona_filename opinions_qa/persona/date0729_midterm_personas_full_haiku.json

# python -m experiment.predict \
#     --exp_setting vanilla_no_history_persona \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name date0730_vanilla_no_history_persona_v4low_full_sonnetinfer_American_Trends_Panel_W26.json \
#     --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json

# python -m experiment.predict \
#     --use_only_relevant_persona \
#     --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
#     --exp_setting vanilla_no_history_persona \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name date0731_vanilla_no_history_persona_v5selected_full_sonnetinfer_American_Trends_Panel_W26.json \
#     --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json

# python -m experiment.predict \
#     --use_only_relevant_persona \
#     --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
#     --exp_setting vanilla_no_history_persona \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name date0731_vanilla_no_history_persona_v5low_selected_sonnetinfer_American_Trends_Panel_W26.json \
#     --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json

python -m experiment.predict \
    --use_only_relevant_persona \
    --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
    --exp_setting vanilla_no_history_persona \
    --survey_name American_Trends_Panel_W26 \
    --log_name date0801_vanilla_no_history_persona_v5low_selected_sonnetinfer_random_cand_American_Trends_Panel_W26.json \
    --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json