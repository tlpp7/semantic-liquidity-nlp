import argparse
import json
from semantic_liquidity.catalog import import_csv, import_huggingface, list_datasets

p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="command",required=True)
csv = sub.add_parser("csv")
csv.add_argument("path")
csv.add_argument("--text-col",default="text")
csv.add_argument("--label-col",default="target")
csv.add_argument("--license",default="unknown")
hf = sub.add_parser("hf")
hf.add_argument("dataset_id")
hf.add_argument("--split",default="train")
hf.add_argument("--text-col",default="text")
hf.add_argument("--label-col",default="label")
hf.add_argument("--revision")
hf.add_argument("--license",default="unknown")
sub.add_parser("list")
a=p.parse_args()
if a.command=="csv":
    result=import_csv(a.path,text_col=a.text_col,label_col=a.label_col,license_name=a.license)
elif a.command=="hf":
    result=import_huggingface(a.dataset_id,a.split,a.text_col,a.label_col,revision=a.revision,license_name=a.license)
else:
    result=list_datasets()
print(json.dumps(result,indent=2,ensure_ascii=False))
