from __future__ import annotations

import argparse,re

from pathlib import Path

import pandas as pd

def parse_param(setting,name):

    pats={'score':r'score_([0-9p]+)','avg':r'_avg([0-9p]+)','m':r'_m([0-9]+)_','cov':r'_cov([0-9p]+)','k':r'_k([0-9a-zA-Z]+)'}; m=re.search(pats[name],setting)

    if not m: return None

    v=m.group(1); return v if name=='k' else float(v.replace('p','.'))

def read_stats(path):

    try:

        df=pd.read_csv(path); return {} if df.empty else df.iloc[0].to_dict()

    except Exception: return {}

def get_stat(d,*keys):

    for k in keys:

        if k in d and not pd.isna(d[k]): return d[k]

    return None

def infer(root,path):

    try: parts=path.relative_to(root).parts

    except Exception: return None

    return (parts[0],parts[2]) if len(parts)>=4 and parts[1]=='experiments' else None

def main():

    ap=argparse.ArgumentParser(); ap.add_argument('--out_root',required=True); ap.add_argument('--eval_glob',default='eval_percol_faiss_only*.csv'); ap.add_argument('--ks',nargs='+',type=int,default=[10,60]); ap.add_argument('--summary_csv',required=True); ap.add_argument('--compact_csv',required=True); ap.add_argument('--make_grid_pivots',action='store_true'); ap.add_argument('--pivot_prefix',default='pivot'); args=ap.parse_args(); root=Path(args.out_root); rows=[]; allowed=set(args.ks)

    for ev in sorted(root.rglob(args.eval_glob)):

        inf=infer(root,ev)

        if not inf: continue

        sweep,setting=inf; stats=read_stats(root/sweep/'relationmaps'/setting/'filter_stats.csv')

        try: df=pd.read_csv(ev)

        except Exception as e: print('Could not read',ev,e); continue

        if 'k' not in df.columns or 'map' not in df.columns: continue

        for k,g in df.groupby('k'):

            k=int(k)

            if k not in allowed: continue

            sort_cols=['map','recall'] if 'recall' in g.columns else ['map']; best=g.sort_values(sort_cols,ascending=False).iloc[0]

            rows.append({'sweep':sweep,'setting':setting,'param_semantic_score':parse_param(setting,'score'),'param_avg_match':parse_param(setting,'avg'),'param_m_min':parse_param(setting,'m'),'param_coverage_min':parse_param(setting,'cov'),'param_k':parse_param(setting,'k'),'k':k,'map':float(best['map']),'recall':float(best['recall']) if 'recall' in best else None,'best_epoch':int(best['epoch']) if 'epoch' in best and not pd.isna(best['epoch']) else None,'checkpoint':best['checkpoint'] if 'checkpoint' in best else '','eval_csv':str(ev),'relation_map':str(root/sweep/'relationmaps'/setting/'relation_map_semantic.json'),'filter_stats_csv':str(root/sweep/'relationmaps'/setting/'filter_stats.csv'),'anchors_with_positives':get_stat(stats,'after_anchors','anchors_with_positives'),'directed_edges':get_stat(stats,'after_directed_edges','directed_edges'),'avg_degree':get_stat(stats,'after_avg_degree','avg_degree'),'median_degree':get_stat(stats,'after_median_degree','median_degree'),'max_degree':get_stat(stats,'after_max_degree','max_degree'),'semantic_score_mean':get_stat(stats,'after_score_mean','semantic_score_mean'),'avg_match_mean':get_stat(stats,'after_avg_match_mean','avg_match_mean'),'matched_count_mean':get_stat(stats,'after_matched_count_mean','matched_count_mean'),'coverage_mean':get_stat(stats,'after_coverage_min_mean','coverage_min_mean')})

    summary=pd.DataFrame(rows)

    if not summary.empty: summary=summary.sort_values(['k','map'],ascending=[True,False])

    Path(args.summary_csv).parent.mkdir(parents=True,exist_ok=True); summary.to_csv(args.summary_csv,index=False)

    comp=[]

    if not summary.empty:

        for (sweep,setting),sub in summary.groupby(['sweep','setting'],dropna=False):

            row={'sweep':sweep,'setting':setting}

            for k in args.ks:

                g=sub[sub['k']==k]

                if not g.empty:

                    b=g.sort_values(['map','recall'],ascending=False).iloc[0]; row[f'map@{k}']=b['map']; row[f'recall@{k}']=b['recall']; row[f'best_epoch@{k}']=b['best_epoch']

            for c in ['param_semantic_score','param_avg_match','param_m_min','param_coverage_min','param_k','directed_edges','avg_degree','matched_count_mean','coverage_mean']:

                if c in sub.columns: row[c]=sub[c].iloc[0]

            comp.append(row)

    compact=pd.DataFrame(comp); Path(args.compact_csv).parent.mkdir(parents=True,exist_ok=True); compact.to_csv(args.compact_csv,index=False)

    print('Saved summary:',args.summary_csv,'rows=',len(summary)); print('Saved compact:',args.compact_csv,'settings=',len(compact))

    if args.make_grid_pivots and not summary.empty:

        grid=summary[summary['sweep']=='grid2d_score_coverage_sweep']

        for k in args.ks:

            g=grid[grid['k']==k]

            if not g.empty:

                out=root/f'{args.pivot_prefix}_grid2d_map{k}_score_x_coverage.csv'; g.pivot_table(index='param_semantic_score',columns='param_coverage_min',values='map',aggfunc='max').to_csv(out); print('Saved pivot:',out)

if __name__=='__main__': main()
