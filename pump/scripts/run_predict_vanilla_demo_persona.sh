# python -m experiment.predict \
#     --exp_setting vanilla_demo_persona \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name date0729_vanilla_demo_persona_v3high_American_Trends_Panel_W26.json \
#     --persona_filename opinions_qa/persona/date0729_midterm_personas.json

# python -m experiment.predict \
#     --exp_setting vanilla_demo_persona \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name date0730_vanilla_demo_persona_v4low_full_haikuinfer_American_Trends_Panel_W26.json \
#     --persona_filename opinions_qa/persona/date0729_midterm_personas_full_haiku.json

# python -m experiment.predict \
#     --exp_setting vanilla_demo_persona \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name date0730_vanilla_demo_persona_v4low_full_sonnetinfer_American_Trends_Panel_W26.json \
#     --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json

# python -m experiment.predict \
#     --use_only_relevant_persona \
#     --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
#     --exp_setting vanilla_demo_persona \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name date0731_vanilla_demo_persona_v5selected_full_sonnetinfer_American_Trends_Panel_W26.json \
#     --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json

# python -m experiment.predict \
#     --use_only_relevant_persona \
#     --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
#     --exp_setting vanilla_demo_persona \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name date0731_vanilla_demo_persona_v5low_selected_sonnetinfer_random_cand_American_Trends_Panel_W26.json \
#     --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json

# python -m experiment.predict_review \
#     --use_only_relevant_persona \
#     --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
#     --exp_setting vanilla_demo_persona \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name date0731_vanilla_demo_persona_v5low_selected_sonnetinfer_American_Trends_Panel_W26.json \
#     --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json


############### 0802

# python -m experiment.predict \
#     --use_only_relevant_persona \
#     --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
#     --exp_setting vanilla_demo_persona \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name date0802_vanilla_demo_persona_v6all_namevalue_selected_sonnetinfer_American_Trends_Panel_W26.json \
#     --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json

# python -m experiment.predict \
#     --use_only_relevant_persona \
#     --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
#     --exp_setting vanilla_demo_persona \
#     --survey_name American_Trends_Panel_W26 \
#     --log_name date0802_vanilla_demo_persona_v6low_namevalue_selected_sonnetinfer_American_Trends_Panel_W26.json \
#     --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json


############### 0802 v7

python -m experiment.predict \
    --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
    --exp_setting vanilla_demo_persona \
    --survey_name American_Trends_Panel_W26 \
    --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json \
    --persona_level low mid high \
    --persona_repr namevalue \
    --log_name date0802_v7_vanilla_demo_persona_alllevels_allpersonas_namevalue_sonnetinfer_American_Trends_Panel_W26.json


python -m experiment.predict \
    --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
    --exp_setting vanilla_demo_persona \
    --survey_name American_Trends_Panel_W26 \
    --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json \
    --persona_level low mid high \
    --persona_repr descvalue \
    --log_name date0802_v7_vanilla_demo_persona_alllevels_allpersonas_descvalue_sonnetinfer_American_Trends_Panel_W26.json


python -m experiment.predict \
    --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
    --exp_setting vanilla_demo_persona \
    --survey_name American_Trends_Panel_W26 \
    --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json \
    --persona_level low mid high \
    --persona_repr namedescvalue \
    --log_name date0802_v7_vanilla_demo_persona_alllevels_allpersonas_namedescvalue_sonnetinfer_American_Trends_Panel_W26.json

############### 0804 v7

python -m experiment.predict \
    --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
    --exp_setting vanilla_demo_persona \
    --survey_name American_Trends_Panel_W26 \
    --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json \
    --persona_level low mid high \
    --persona_repr namedesccandvalue \
    --log_name date0804_v7_vanilla_demo_persona_alllevels_allpersonas_namedesccandvalue_sonnetinfer_American_Trends_Panel_W26.json


