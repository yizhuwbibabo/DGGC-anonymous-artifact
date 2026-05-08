from __future__ import annotations

import argparse,csv,json,math,statistics

from collections import defaultdict

from pathlib import Path

from typing import Any,Dict,List

TARGET_KEYS=("table","target","neighbor","candidate","dst","to","file","filename","name")

def fnum(x,default=0.0):

    try:

        v=float(x); return default if math.isnan(v) else v

    except Exception: return default

def inum(x,default=0):

    try: return int(float(x))

    except Exception: return default

def family_id(name):

    s=Path(str(name)).stem

    return s.split('____')[0] if '____' in s else (s.split('__')[0] if '__' in s else s)

def infer_target(edge):

    if isinstance(edge,str): return edge

    if isinstance(edge,dict):

        for k in TARGET_KEYS:

            if k in edge and edge[k] is not None: return str(edge[k])

    return None

def normalize_edge(anchor,target,payload):

    d=dict(payload) if isinstance(payload,dict) else {}; d['table']=str(target); d['anchor']=str(anchor)

    aliases={'semantic_score':['score','positive_score','strength','weight','f'],'avg_match':['avg_match_score','average_match','mean_match'],'matched_count':['matched_cols','match_count','num_matched','matched_num'],'coverage_min':['coverage','min_coverage','coverage_query','coverage_q'],'coverage_max':['max_coverage','coverage_candidate','coverage_c']}

    for k,als in aliases.items():

        if k not in d:

            for a in als:

                if a in d: d[k]=d[a]; break

    d['semantic_score']=fnum(d.get('semantic_score'),1.0); d['avg_match']=fnum(d.get('avg_match'),1.0)

    d['matched_count']=inum(d.get('matched_count'),0); d['coverage_min']=fnum(d.get('coverage_min'),1.0)

    d['coverage_max']=fnum(d.get('coverage_max'),d['coverage_min'])

    return d

def normalize_map(raw:Dict[str,Any])->Dict[str,List[Dict[str,Any]]]:

    out={}

    for anchor,value in raw.items():

        edges=[]

        if isinstance(value,dict):

            for target,payload in value.items(): edges.append(normalize_edge(anchor,infer_target(payload) or target,payload))

        elif isinstance(value,list):

            for item in value:

                target=infer_target(item)

                if target: edges.append(normalize_edge(anchor,target,item))

        best={}

        for e in edges:

            t=str(e['table']); key=(fnum(e['semantic_score']),fnum(e['avg_match']),inum(e['matched_count']),fnum(e['coverage_min']))

            old=best.get(t); old_key=(-1,-1,-1,-1) if old is None else (fnum(old['semantic_score']),fnum(old['avg_match']),inum(old['matched_count']),fnum(old['coverage_min']))

            if key>old_key: best[t]=e

        out[str(anchor)]=list(best.values())

    return out

def graph_stats(mp,prefix):

    degrees=[len(v) for v in mp.values() if v]; edges=[e for es in mp.values() for e in es]

    mean=lambda xs: float(statistics.mean(xs)) if xs else 0.0; median=lambda xs: float(statistics.median(xs)) if xs else 0.0

    return {f'{prefix}_anchors_total':len(mp),f'{prefix}_anchors':len(degrees),f'{prefix}_directed_edges':len(edges),f'{prefix}_avg_degree':mean(degrees),f'{prefix}_median_degree':median(degrees),f'{prefix}_max_degree':max(degrees) if degrees else 0,f'{prefix}_score_mean':mean([fnum(e.get('semantic_score')) for e in edges]),f'{prefix}_avg_match_mean':mean([fnum(e.get('avg_match')) for e in edges]),f'{prefix}_matched_count_mean':mean([fnum(e.get('matched_count')) for e in edges]),f'{prefix}_coverage_min_mean':mean([fnum(e.get('coverage_min')) for e in edges]),f'{prefix}_coverage_max_mean':mean([fnum(e.get('coverage_max')) for e in edges])}

def symmetrize(mp):

    out={a:[dict(e) for e in es] for a,es in mp.items()}

    for a,es in list(mp.items()):

        for e in es:

            t=str(e['table']); r=dict(e); r['table']=a; r['anchor']=t; out.setdefault(t,[])

            if not any(str(x.get('table'))==a for x in out[t]): out[t].append(r)

    return out

def filter_map(mp,args):

    out={}

    for a,es in mp.items():

        keep=[e for e in es if fnum(e.get('semantic_score'))>=args.min_score and fnum(e.get('avg_match'))>=args.min_avg_match and inum(e.get('matched_count'))>=args.min_matched_count and fnum(e.get('coverage_min'))>=args.min_coverage_min and fnum(e.get('coverage_max'))>=args.min_coverage_max]

        keep.sort(key=lambda e:(fnum(e['semantic_score']),fnum(e['avg_match']),inum(e['matched_count']),fnum(e['coverage_min'])),reverse=True)

        if args.max_same_family and args.max_same_family>0:

            counts=defaultdict(int); tmp=[]

            for e in keep:

                fam=family_id(e['table'])

                if counts[fam]<args.max_same_family: tmp.append(e); counts[fam]+=1

            keep=tmp

        if args.max_neighbors and args.max_neighbors>0: keep=keep[:args.max_neighbors]

        if keep or not args.drop_empty_anchors: out[a]=keep

    return symmetrize(out) if args.symmetrize else out

def main():

    p=argparse.ArgumentParser(); p.add_argument('--input_map',required=True); p.add_argument('--output_map',required=True); p.add_argument('--output_stats_csv',required=True); p.add_argument('--min_score',type=float,default=0.0); p.add_argument('--min_avg_match',type=float,default=0.0); p.add_argument('--min_matched_count',type=int,default=0); p.add_argument('--min_coverage_min',type=float,default=0.0); p.add_argument('--min_coverage_max',type=float,default=0.0); p.add_argument('--max_neighbors',type=int); p.add_argument('--max_same_family',type=int,default=0); p.add_argument('--symmetrize',action='store_true'); p.add_argument('--drop_empty_anchors',action='store_true'); args=p.parse_args()

    raw=json.loads(Path(args.input_map).read_text(encoding='utf-8')); mp=normalize_map(raw); before=graph_stats(mp,'before'); out=filter_map(mp,args); after=graph_stats(out,'after'); stats={**before,**after}

    Path(args.output_map).parent.mkdir(parents=True,exist_ok=True); Path(args.output_map).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')

    Path(args.output_stats_csv).parent.mkdir(parents=True,exist_ok=True)

    with open(args.output_stats_csv,'w',newline='',encoding='utf-8') as f: writer=csv.DictWriter(f,fieldnames=list(stats.keys())); writer.writeheader(); writer.writerow(stats)

    print('Saved filtered relation_map:',args.output_map); print('Saved stats:',args.output_stats_csv); [print(f'{k}={v}') for k,v in stats.items()]

if __name__=='__main__': main()
