import json
import bnlearn as bn
import pandas as pd
import copy
import argparse
import random
from tqdm import tqdm


def verify(data):
    ok_cnt = 0
    wrong_cnt = 0
    for user in data:
        for persona in data[user]:
            # if all(_ not in persona['candidate_values'] for _ in [persona['inferred_value'], persona['inferred_value'][1:-1]]):
            if persona['inferred_value'] not in persona['candidate_values']:
                wrong_cnt += 1
                print(persona)
                print(f"|{persona['inferred_value']}|")
                print(persona['candidate_values'])
                print()
            else:
                ok_cnt += 1
    print(ok_cnt, wrong_cnt)


# clean data
def clean(data):
    temp_data = copy.deepcopy(data)
    for user in data.keys():
        personas = data[user]
        clean = []
        seen_personas = []
        for p in personas:
            if f"{p['level']}/{p['name']}" in seen_personas:
                continue
            inf_val = p['inferred_value']
            cands = p['candidate_values']
            # if len(cands) > 5:
            #     continue
            if inf_val not in cands:
                if inf_val[1:-1] in cands and all(_ in ['\'', '\"'] for _ in [inf_val[0], inf_val[-1]]):
                    p['inferred_value'] = inf_val[1:-1]
                    clean.append(p)
                    seen_personas.append(f"{p['level']}/{p['name']}")
            else:
                clean.append(p)
                seen_personas.append(f"{p['level']}/{p['name']}")
        temp_data[user] = clean
    return temp_data



def categorize(data, opt2idx_mapping):
    categorized_data = copy.deepcopy(data)
    for user in data:
        personas = data[user]
        clean = []
        for idx, p in enumerate(personas):
            p_name = f"{p['level']}/{p['name']}"
            inf_val = p['inferred_value']
            try:
                p['inferred_value'] = opt2idx_mapping[p_name][inf_val]
            except:
                print('Categorizing issues')
                print(user, p_name, idx, inf_val)
                print(p)
            clean.append(p)
        categorized_data[user] = clean
    return categorized_data



def get_bn(df, method, score):

    model = bn.structure_learning.fit(df, methodtype=method, scoretype=score, tabu_length=100)
    model1 = bn.independence_test(model, df, alpha=0.05, prune=True)
    param_model = bn.parameter_learning.fit(model1, df)

    return param_model


def main(args):
    # load data
    with open(args.persona_filename, 'r') as f:
        data = json.load(f)
    data[list(data.keys())[0]]
    verify(data)

    # cleaning the data
    cleaned_data = clean(data)
    verify(cleaned_data)
    for user in cleaned_data:
        for idx, p in enumerate(cleaned_data[user]):
            if p['inferred_value'] not in p['candidate_values']:
                print(user, idx, p['name'], p['inferred_value'], p['candidate_values'], p)

    print("Number of unique persona dimensions:")
    print(len(set([f"{p['level']}/{p['name']}" for user in cleaned_data for p in cleaned_data[user]])))

    # categorize_data
    idx2opt_mapping = {}
    opt2idx_mapping = {}
    for user in cleaned_data:
        for persona in cleaned_data[user]:
            p_name = f"{persona['level']}/{persona['name']}"
            if p_name in idx2opt_mapping:
                continue
            cands = persona['candidate_values']
            idx2opt = {k+1: v for k, v in enumerate(cands)}  # 0 for potential NAs
            idx2opt[0] = "Unknown"
            idx2opt_mapping[p_name] = idx2opt
            opt2idx_mapping[p_name] = {v: k for k, v in idx2opt.items()}
    categorized_data = categorize(cleaned_data, opt2idx_mapping)

    # put into df
    res = []
    for user in categorized_data.keys():
        entry = {'user': user}
        for persona in categorized_data[user]:
            entry[f"{persona['level']}/{persona['name']}"] = persona['inferred_value']
        res.append(entry)
    raw_df = pd.DataFrame(res)
    raw_df.fillna(0, inplace=True)
    raw_df = raw_df.astype(int)
    df = raw_df[[_ for _ in raw_df.columns if _ != "user"]]
    if len(set(df.columns)) > 100:
        df = df[random.sample(list(df.columns), k=100)]

    # update the data
    for _ in range(args.em_k):
        param_model = get_bn(df, method=args.method, score=args.score)
        Xtest = bn.sampling(param_model, 10)
        cols = Xtest.columns
        variables=[col for col in cols if col.startswith('mid') or col.startswith('high')]
        for var in tqdm(variables):
            Pout = bn.predict(param_model, df[[col for col in cols if col not in variables]], variables=[var])
            df[var] = Pout[var]
            raw_df[var] = Pout[var]

    # save the updated data
    for _, row in raw_df.iterrows():
        user = str(row['user'])
        personas = categorized_data[user]
        for idx, p in enumerate(personas):
            val = row[f"{p['level']}/{p['name']}"]
            mapping = idx2opt_mapping[f"{p['level']}/{p['name']}"]
            new_val = mapping[val]
            categorized_data[user][idx]['inferred_value'] = new_val

    with open(args.new_persona_filename, 'w') as f:
        json.dump(categorized_data, f, indent=4)


if __name__ == '__main__':
    random.seed(42)

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--persona_filename", type=str)
    argparser.add_argument("--new_persona_filename", type=str)
    argparser.add_argument("--method", type=str)
    argparser.add_argument("--score", type=str)
    argparser.add_argument("--em_k", type=int)

    args = argparser.parse_args()
    main(args)