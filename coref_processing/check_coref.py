import json

"""
Xiulin's function to check the coreference clusters
"""

def read_jsonlines(jsonfile):
    jsonlist =[]
    with open(jsonfile, 'r') as f:
        for line in f:
            jsonlist.append(json.loads(line))
    return jsonlist


if __name__=='__main__':
    final = []
    pred = read_jsonlines('./ara_pronoun_pred.jsonlines')
    for recipe in pred:
        final_clusters = {}
        clusters_token = []
        tokens = [x for y in recipe['sentences'] for x in y]
        for cluster in recipe['predicted_clusters']:
            cluster_span = []
            for span in cluster:
                token_span = ' '.join(tokens[span[0]:span[-1]+1])
                cluster_span.append(token_span)
            clusters_token.append(cluster_span)
        final_clusters['recipe_id'] = recipe['recipe_id']
        final_clusters['clusters'] = clusters_token
        final_clusters['sentences'] = ' '.join([' '.join(x) for x in recipe['sentences']])
        final.append(final_clusters)

    for f in final:
        clust = f['clusters']
        print(f['recipe_id'])
        print(clust)
        #for cl in clust:
            #stripped_clust = [c for c in cl if c not in ['it', 'them']]
            #print(stripped_clust)
            #if len(stripped_clust) > 1:
                #print("Found")
