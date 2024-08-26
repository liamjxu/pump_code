# # history
# for i in 1
# do
# python -m src.predict \
#     --exp_setting history \
#     --survey_name American_Trends_Panel_W26 \
#     --persona_level low mid high \
#     --persona_repr namedescvalue \
#     --log_name date0825_v10post_midterm_history_American_Trends_Panel_W26_run${i}.json \
#     --persona_filename opinions_qa/persona_val/date0729_midterm_personas_full_sonnet.json
# done

# # demo
# for i in 1
# do
# python -m src.predict \
#     --exp_setting demo \
#     --survey_name American_Trends_Panel_W26 \
#     --persona_level low mid high \
#     --persona_repr namedescvalue \
#     --log_name date0825_v10post_midterm_demo_American_Trends_Panel_W26_run${i}.json \
#     --persona_filename opinions_qa/persona_val/date0729_midterm_personas_full_sonnet.json
# done

# # history + demo
# for i in 1
# do
# python -m src.predict \
#     --exp_setting history_demo \
#     --survey_name American_Trends_Panel_W26 \
#     --persona_level low mid high \
#     --persona_repr namedescvalue \
#     --log_name date0825_v10post_midterm_history_demo_American_Trends_Panel_W26_run${i}.json \
#     --persona_filename opinions_qa/persona_val/date0729_midterm_personas_full_sonnet.json
# done

# persona
for i in 1
do
python -m src.predict \
    --exp_setting persona \
    --survey_name American_Trends_Panel_W26 \
    --persona_level low mid high \
    --persona_repr namedescvalue \
    --log_name date0825_v10post_midterm_persona_American_Trends_Panel_W26_run${i}.json \
    --persona_filename opinions_qa/persona_val/date0729_midterm_personas_full_sonnet.json
done

# history + persona
for i in 1
do
python -m src.predict \
    --exp_setting history_persona \
    --survey_name American_Trends_Panel_W26 \
    --persona_level low mid high \
    --persona_repr namedescvalue \
    --log_name date0825_v10post_midterm_history_persona_American_Trends_Panel_W26_run${i}.json \
    --persona_filename opinions_qa/persona_val/date0729_midterm_personas_full_sonnet.json
done

# demo + persona
for i in 1
do
python -m src.predict \
    --exp_setting demo_persona \
    --survey_name American_Trends_Panel_W26 \
    --persona_level low mid high \
    --persona_repr namedescvalue \
    --log_name date0825_v10post_midterm_demo_persona_American_Trends_Panel_W26_run${i}.json \
    --persona_filename opinions_qa/persona_val/date0729_midterm_personas_full_sonnet.json
done


# history + demo + persona
for i in 1
do
python -m src.predict \
    --exp_setting history_demo_persona \
    --survey_name American_Trends_Panel_W26 \
    --persona_level low mid high \
    --persona_repr namedescvalue \
    --log_name date0825_v10post_midterm_history_demo_persona_American_Trends_Panel_W26_run${i}.json \
    --persona_filename opinions_qa/persona_val/date0729_midterm_personas_full_sonnet.json
done
