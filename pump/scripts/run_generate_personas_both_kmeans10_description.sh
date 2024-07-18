python -m generate_personas \
    --debug \
    --output_dir_root sm_local/outputs_sonnet_kmeans10_single_description \
    --model_id anthropic.claude-3-sonnet-20240229-v1:0 \
    --extraction_prompt_type description \
    --clustering_algo kmeans \
    --clustering_num_clusters 10 \
    --merging_personas_from_surveys single \
    --survey_ending 5 \
    --phases summarizing cleaning 

python -m generate_personas \
    --debug \
    --output_dir_root sm_local/outputs_haiku_kmeans10_single_description \
    --model_id anthropic.claude-3-haiku-20240307-v1:0 \
    --extraction_prompt_type description \
    --clustering_algo kmeans \
    --clustering_num_clusters 10 \
    --merging_personas_from_surveys single \
    --survey_ending 5 \
    --phases summarizing cleaning 
