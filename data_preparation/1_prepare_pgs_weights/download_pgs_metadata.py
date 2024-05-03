import json
import urllib.request
import pandas as pd
import glob
import os.path as osp

metadata = []
pgs_ids = [osp.basename(f).replace(".txt.gz", "") 
           for f in glob.glob("data/pgs_weights/EFO_0004339/*.txt.gz")]

for pgs_id in pgs_ids:

    print(pgs_id)
    
    info = json.loads(urllib.request.urlopen(f"https://www.pgscatalog.org/rest/score/{pgs_id}").read())
    
    metadata.append({
        'PGS_ID': pgs_id,
        'name': info['name'],
        'publication': info['publication']['title'],
        'method': info['method_name']
    })

pgs_metadata = pd.DataFrame(metadata)
pgs_metadata.to_csv("metadata/pgs_weights.txt", index=False)

