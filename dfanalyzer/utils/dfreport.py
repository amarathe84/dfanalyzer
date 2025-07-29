#!/usr/bin/env python3
"""
Command-line utility to summarize dftracer .pfw.gz traces.
Summaries are per-node, per-process, per-thread, or individual events.
Use --all-events with --node, --process, or --thread to output event-level summaries instead of grouped categories.
Add --aggregate to highlight the node with the max of each metric across nodes.

Usage examples:
  ./dfanalyze_v0.06.py --node /path/to/COMPACT/
  ./dfanalyze_v0.06.py --node --all-events /path/to/COMPACT/
  ./dfanalyze_v0.06.py --process /path/to/COMPACT/
  ./dfanalyze_v0.06.py --process --all-events /path/to/COMPACT/
  ./dfanalyze_v0.06.py --thread /path/to/COMPACT/
  ./dfanalyze_v0.06.py --thread --all-events /path/to/COMPACT/
  ./dfanalyze_v0.06.py --events /path/to/COMPACT/
  ./dfanalyze_v0.06.py --node --aggregate /path/to/COMPACT/

Options are mutually exclusive, one of --node, --process, --thread, or --events is required.
"""
import argparse
import os
import glob
import gzip
import json
from collections import defaultdict
import pandas as pd
from colorama import Fore, Style, init as colorama_init

colorama_init()

#--------------- Load & Summaries ----------------

def assign_group(name: str) -> str:
    nl = name.lower()
    if nl.startswith('torchframework'):     return 'TorchFramework'
    if nl.startswith('pytorchdataloader'):  return 'PytorchDataLoader'
    if nl.startswith('pytorchcheckpointing'):return 'PytorchCheckpointing'
    if nl.startswith('filestorage'):        return 'FileStorage'
    if nl.startswith('dlio'):               return 'DLIO'
    if any(tok in nl for tok in ('open','close','start')): return 'file_ops'
    if any(tok in nl for tok in ('read','seek')):          return 'read_seek'
    if 'loop' in nl:                  return 'loop'
    if 'stat' in nl or 'xstat' in nl: return 'attr_checks'
    if 'npz' in nl:                   return 'npz_ops'
    return 'other'


def load_node_df(node_dir: str) -> pd.DataFrame|None:
    records=[]
    files=glob.glob(os.path.join(node_dir,'*.pfw.gz'))
    compact=os.path.join(node_dir,'COMPACT')
    if os.path.isdir(compact):
        files+=glob.glob(os.path.join(compact,'*.pfw.gz'))
    for p in sorted(files):
        with gzip.open(p,'rt') as f:
            for raw in f:
                line=raw.strip().rstrip(',')
                if not line or line in ('[',']'): continue
                try:
                    obj=json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(obj)
    if not records: return None
    df=pd.json_normalize(records)
    df['name']=df.get('name','').astype(str)
    df['dur']=pd.to_numeric(df.get('dur',0),errors='coerce').fillna(0.0)
    df['pid']=pd.to_numeric(df.get('pid',0),errors='coerce').fillna(0).astype(int)
    df['tid']=pd.to_numeric(df.get('tid',0),errors='coerce').fillna(0).astype(int)
    return df[['name','dur','pid','tid']]


def summarize_groups(df):
    df=df[df['dur']>0].copy()
    df['group']=df['name'].apply(assign_group)
    total=df['dur'].sum()
    agg=(df.groupby('group')['dur']
        .agg(Total_Time='sum',Num_Instances='count',Average='mean',Min='min',Max='max',StdDev='std')
        .reset_index())
    agg['% Total Time']=(100*agg['Total_Time']/total).round(3)
    for c in ['Total_Time','Average','Min','Max','StdDev']:
        agg[c]=agg[c].round(3)
    return agg.sort_values('% Total Time',ascending=False)[['group','% Total Time','Total_Time','Num_Instances','Average','Min','Max','StdDev']]


def summarize_events(df):
    df=df[df['dur']>0]
    total=df['dur'].sum()
    agg=(df.groupby('name')['dur']
        .agg(Total_Time='sum',Num_Instances='count',Average='mean',Min='min',Max='max',StdDev='std')
        .reset_index())
    agg['% Total Time']=(100*agg['Total_Time']/total).round(3)
    for c in ['Total_Time','Average','Min','Max','StdDev']:
        agg[c]=agg[c].round(3)
    return agg.sort_values('% Total Time',ascending=False)[['name','% Total Time','Total_Time','Num_Instances','Average','Min','Max','StdDev']]


def summarize_process_groups(df):
    df=df[df['dur']>0].copy()
    df['group']=df['name'].apply(assign_group)
    agg=(df.groupby(['pid','group'])['dur']
        .agg(Total_Time='sum',Num_Instances='count',Average='mean',Min='min',Max='max',StdDev='std')
        .reset_index())
    pct=(agg.groupby('pid')['Total_Time'].sum().rename('pid_total').reset_index())
    agg=agg.merge(pct,on='pid')
    agg['% Total Time']=(100*agg['Total_Time']/agg['pid_total']).round(3)
    for c in ['Total_Time','Average','Min','Max','StdDev']:
        agg[c]=agg[c].round(3)
    return agg.sort_values(['pid','% Total Time'],ascending=[True,False])[['pid','group','% Total Time','Total_Time','Num_Instances','Average','Min','Max','StdDev']]


def summarize_thread_groups(df):
    df=df[df['dur']>0].copy()
    df['group']=df['name'].apply(assign_group)
    agg=(df.groupby(['tid','group'])['dur']
        .agg(Total_Time='sum',Num_Instances='count',Average='mean',Min='min',Max='max',StdDev='std')
        .reset_index())
    pct=(agg.groupby('tid')['Total_Time'].sum().rename('tid_total').reset_index())
    agg=agg.merge(pct,on='tid')
    agg['% Total Time']=(100*agg['Total_Time']/agg['tid_total']).round(3)
    for c in ['Total_Time','Average','Min','Max','StdDev']:
        agg[c]=agg[c].round(3)
    return agg.sort_values(['tid','% Total Time'],ascending=[True,False])[['tid','group','% Total Time','Total_Time','Num_Instances','Average','Min','Max','StdDev']]


def build_group_map(df):
    gm=defaultdict(set)
    for nm in df['name'].unique(): gm[assign_group(nm)].add(nm)
    return {g:sorted(list(ev)) for g,ev in gm.items()}


def print_tree_for_node(node,df,group_map,mode,all_ev):
    print(f"\n===== Summary for {node} =====\n")
    if mode=='events':
        print(summarize_events(df).to_string(index=False)); return
    if mode=='node':
        print((summarize_events(df) if all_ev else summarize_groups(df)).to_string(index=False));return
    if mode=='process':
        if all_ev:
            for pid,sub in df.groupby('pid'):
                print(f"--- Process {pid} ---")
                print(summarize_events(sub).to_string(index=False));print("============")
        else:
            for pid,sub in summarize_process_groups(df).groupby('pid'):
                print(f"--- Process {pid} ---")
                print(sub.drop(columns='pid').to_string(index=False));print("============")
        return
    if mode=='thread':
        if all_ev:
            for tid,sub in df.groupby('tid'):
                print(f"--- Thread {tid} ---")
                print(summarize_events(sub).to_string(index=False));print("-------")
        else:
            for tid,sub in summarize_thread_groups(df).groupby('tid'):
                print(f"--- Thread {tid} ---")
                print(sub.drop(columns='tid').to_string(index=False));print("-------")
        return


def highlight_across_nodes(per_node, key_col,metrics):
    nodes=list(per_node.keys())
    keys=sorted({k for df in per_node.values() for k in df[key_col].tolist()})
    for key in keys:
        print(f"\n>>> {key}")
        max_holder={m: max(((n,per_node[n].set_index(key_col).get(m,0).get(key,0)) for n in nodes),key=lambda x:x[1])[0]
                    for m in metrics}
        hdr="    node"+"".join(m.rjust(12) for m in metrics)
        print(hdr)
        for n in nodes:
            row=per_node[n]
            vals={m:row.set_index(key_col).get(m,0).get(key,0) for m in metrics}
            line="    "+n.ljust(11)
            for m in metrics:
                col=Fore.RED if n==max_holder[m] else Fore.GREEN
                line+=col+f"{vals[m]:12.3f}"+Style.RESET_ALL
            print(line)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('directory')
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--node',action='store_true')
    g.add_argument('--process',action='store_true')
    g.add_argument('--thread',action='store_true')
    g.add_argument('--events',action='store_true')
    p.add_argument('--all-events',action='store_true')
    p.add_argument('--aggregate',action='store_true',help='Highlight max metric across nodes')
    args=p.parse_args()
    base=args.directory
    if not os.path.isdir(base): p.error(f"{base} not a dir")
    if glob.glob(os.path.join(base,'*.pfw.gz')): nodes=[base]
    else: nodes=[os.path.join(base,d) for d in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base,d))]
    raw={}
    for nd in nodes:
        df=load_node_df(nd)
        if df is None: continue
        nm=os.path.basename(nd.rstrip(os.sep))
        raw[nm if nm.lower()!='compact' else os.path.basename(os.path.dirname(nd))]=df
    if not raw:
        print("No traces");return
    group_map=build_group_map(pd.concat(raw.values(),ignore_index=True))
    mode=('events' if args.events else 'process' if args.process else 'thread' if args.thread else 'node')
    for n, df in raw.items():
        # Choose behavior for process + aggregate: ascii tree with avg only
        if mode == 'process' and args.aggregate:
            # Aggregate per-process: ascii tree by average, highlight group-wise max across all processes
            print(f"\n===== Summary for {n} (aggregate per-process) =====\n")
            proc_df = summarize_process_groups(df)
            # Determine, for each group, which pid has the maximum average
            max_pid_map = proc_df.loc[
                proc_df.groupby('group')['Average'].idxmax()
            ].set_index('group')['pid'].to_dict()
            for pid, sub in proc_df.groupby('pid'):
                print(f"Process {pid}")
                for _, row in sub.iterrows():
                    print("|")
                    # backslash escaped to print literal \___
                    prefix = "\\___"
                    grp = row['group']
                    avg = row['Average']
                    line = f"{prefix} {grp} (avg: {avg:.6e})"
                    # highlight in red if this process holds the max for this group, otherwise green
                    if max_pid_map.get(grp) == pid:
                        print(Fore.RED + line + Style.RESET_ALL)
                    else:
                        print(Fore.GREEN + line + Style.RESET_ALL)
                print()
            continue

        # Standard tree output
        print_tree_for_node(n, df, group_map, mode, args.all_events)

    # Global aggregate highlighting for other modes
    if args.aggregate and not (mode == 'process'):
        summ = {}
        metrics = ['% Total Time','Total_Time','Num_Instances','Average','Min','Max','StdDev']
        for n, df in raw.items():
            if mode=='node':
                summ[n] = summarize_events(df) if args.all_events else summarize_groups(df)
            elif mode=='events':
                summ[n] = summarize_events(df)
            elif mode=='process':
                summ[n] = summarize_events(df) if args.all_events else summarize_process_groups(df)
            else:
                summ[n] = summarize_events(df) if args.all_events else summarize_thread_groups(df)
        key_col = ('name' if mode=='events' else 'group')
        highlight_across_nodes(summ, key_col, metrics)

if __name__=='__main__': main()

