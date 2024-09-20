# Generate with haiku
python -m src.generate_personas \
    --debug \
    --using_personadb_surveys \
    --output_dir_root opinions_qa/persona_dim/date0920_haiku_kmeans10_single_example_known_test_use_all_q \
    --model_name haiku \
    --extraction_prompt_type description \
    --clustering_algo kmeans \
    --clustering_num_clusters 10 \
    --merging_personas_from_surveys single \
    --phases extraction clustering summarizing cleaning