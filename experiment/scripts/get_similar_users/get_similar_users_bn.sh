export DATE=0920
export SURVEY_NAME=American_Trends_Panel_W$1
export PERSONA_VAL_PATH=opinions_qa/persona_val/$SURVEY_NAME/date${DATE}_personas_full_personadb.json

# hyperparameters searched in our experiments
if [ "$score" -eq 34 ]; then
    export SETTING="demo_persona"
    export BN="bds"
    export TOP_K=80
    export SKEW_THRES=10
elif [ "$score" -eq 41 ]; then
    export SETTING="persona_only"
    export BN="bic"
    export TOP_K=160
    export SKEW_THRES=15
elif [ "$score" -eq 82 ]; then
    export SETTING="demo_persona"
    export BN="bdeu"
    export TOP_K=100
    export SKEW_THRES=2
else
    echo "Score is something else."
fi

python -m src.get_user_mapping \
    --using_personadb_surveys \
    --survey_name $SURVEY_NAME \
    --persona_val_path $PERSONA_VAL_PATH \
    --bn $BN \
    --setting $SETTING \
    --top_k $TOP_K \
    --skew_thres $SKEW_THRES
