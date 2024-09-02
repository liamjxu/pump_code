# Generate with haiku
python -m src.generate_personas \
    --debug \
    --output_dir_root opinions_qa/persona_dim/date0901_haiku_kmeans10_single_example_known_test_use_all_q \
    --model_name haiku \
    --extraction_prompt_type example \
    --clustering_algo kmeans \
    --clustering_num_clusters 10 \
    --merging_personas_from_surveys single \
    --survey_starting 4 \
    --survey_ending 5 \
    --phases extraction clustering summarizing cleaning