python -m generate_personas \
    --debug \
    --output_dir_root sm_local/outputs_haiku_dbscan_single_example_test \
    --model_id anthropic.claude-3-haiku-20240307-v1:0 \
    --extraction_prompt_type example \
    --clustering_algo dbscan \
    --clustering_num_clusters 0 \
    --merging_personas_from_surveys single \
    --survey_ending 1
