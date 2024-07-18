python -m generate_personas \
    --debug \
    --output_dir_root sm_local/outputs_sonnet_kmeans20_single_example \
    --model_id anthropic.claude-3-sonnet-20240229-v1:0 \
    --extraction_prompt_type example \
    --clustering_algo kmeans \
    --clustering_num_clusters 20 \
    --merging_personas_from_surveys single \
    --survey_ending 5

python -m generate_personas \
    --debug \
    --output_dir_root sm_local/outputs_haiku_kmeans20_single_example \
    --model_id anthropic.claude-3-haiku-20240307-v1:0 \
    --extraction_prompt_type example \
    --clustering_algo kmeans \
    --clustering_num_clusters 20 \
    --merging_personas_from_surveys single \
    --survey_ending 5
