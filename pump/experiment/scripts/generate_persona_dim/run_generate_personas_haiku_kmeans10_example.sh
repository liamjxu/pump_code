# Generate with haiku
python -m src.generate_personas \
    --debug \
    --output_dir_root opinions_qa/persona_dim/haiku_kmeans10_single_example \
    --model_id anthropic.claude-3-haiku-20240307-v1:0 \
    --extraction_prompt_type example \
    --clustering_algo kmeans \
    --clustering_num_clusters 20 \
    --merging_personas_from_surveys single \
    --survey_ending 5 \
    --phases extraction clustering summarizing cleaning