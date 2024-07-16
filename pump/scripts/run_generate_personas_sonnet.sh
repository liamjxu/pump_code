python -m generate_personas \
    --debug \
    --skip_existing \
    --output_dir_root sm_local/outputs_sonnet_kmeans_single_example \
    --model_id anthropic.claude-3-sonnet-20240229-v1:0 \
    --extraction_prompt_type example \
    --clustering_algo kmeans \
    --clustering_num_clusters 20 \
    --merging_personas_from_surveys single \
    --survey_ending 5