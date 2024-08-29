# Generate with haiku
python -m src.generate_personas \
    --debug \
    --output_dir_root opinions_qa/persona_dim/date0828_haiku_kmeans20_single_example \
    --model_name haiku \
    --extraction_prompt_type example \
    --clustering_algo kmeans \
    --clustering_num_clusters 20 \
    --merging_personas_from_surveys single \
    --survey_starting 1 \
    --survey_ending 5 \
    --phases extraction clustering summarizing cleaning