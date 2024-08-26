############### 0804 v8

python -m experiment.predict \
    --use_cot \
    --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
    --exp_setting vanilla_cot \
    --survey_name American_Trends_Panel_W26 \
    --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json \
    --persona_level low mid high \
    --persona_repr namedescvalue \
    --log_name date0804_v8_vanilla_cot_alllevels_allpersonas_namedescvalue_sonnetinfer_American_Trends_Panel_W26.json


