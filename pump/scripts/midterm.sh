
# history
for i in 1 2 3
do
python -m experiment.predict \
    --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
    --exp_setting vanilla \
    --survey_name American_Trends_Panel_W26 \
    --persona_level low mid high \
    --persona_repr namedescvalue \
    --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json \
    --log_name date0805_v9prepmidterm_vanilla_random_American_Trends_Panel_W26_run${i}_fix.json 
done

# demo
for i in 1 2 3
do
python -m experiment.predict \
    --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
    --exp_setting vanilla_no_history_demo \
    --survey_name American_Trends_Panel_W26 \
    --persona_level low mid high \
    --persona_repr namedescvalue \
    --log_name date0805_v9prepmidterm_vanilla_no_history_demo_random_American_Trends_Panel_W26_run${i}_fix.json \
    --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json
done

# history + demo
for i in 1 2 3
do
python -m experiment.predict \
    --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
    --exp_setting vanilla_demo \
    --survey_name American_Trends_Panel_W26 \
    --persona_level low mid high \
    --persona_repr namedescvalue \
    --log_name date0805_v9prepmidterm_vanilla_demo_random_American_Trends_Panel_W26_run${i}_fix.json \
    --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json
done

# persona
for i in 1 2 3
do
python -m experiment.predict \
    --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
    --exp_setting vanilla_no_history_persona \
    --survey_name American_Trends_Panel_W26 \
    --persona_level low mid high \
    --persona_repr namedescvalue \
    --log_name date0805_v9prepmidterm_vanilla_no_history_persona_random_American_Trends_Panel_W26_run${i}_fix.json \
    --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json
done

# # history + persona
# for i in 1 2 3
# do
# python -m experiment.predict \
#     --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
#     --exp_setting vanilla_persona \
#     --survey_name American_Trends_Panel_W26 \
#     --persona_level low mid high \
#     --persona_repr namedescvalue \
#     --log_name date0805_v9prepmidterm_vanilla_persona_random_American_Trends_Panel_W26_run${i}_fix.json \
#     --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json
# done

# # demo + persona
# for i in 1 2 3
# do
# python -m experiment.predict \
#     --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
#     --exp_setting vanilla_no_history_demo_persona \
#     --survey_name American_Trends_Panel_W26 \
#     --persona_level low mid high \
#     --persona_repr namedescvalue \
#     --log_name date0805_v9prepmidterm_vanilla_no_history_demo_persona_random_American_Trends_Panel_W26_run${i}_fix.json \
#     --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json
# done


# # history + demo + persona
# for i in 1 2 3
# do
# python -m experiment.predict \
#     --query_to_persona_idx_mapping_filename opinions_qa/date0731_midterm_query_persona_idx_mapping.json \
#     --exp_setting vanilla_demo_persona \
#     --survey_name American_Trends_Panel_W26 \
#     --persona_level low mid high \
#     --persona_repr namedescvalue \
#     --log_name date0805_v9prepmidterm_vanilla_demo_persona_random_American_Trends_Panel_W26_run${i}_fix.json \
#     --persona_filename opinions_qa/persona/date0729_midterm_personas_full_sonnet.json
# done
