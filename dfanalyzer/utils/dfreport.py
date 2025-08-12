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
import dask.dataframe as dd
from colorama import Fore, Style, init as colorama_init

# Initialize colorama for terminal colors
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
    # Look for both .pfw.gz and .pfw files
    gz_files = glob.glob(os.path.join(node_dir, '*.pfw.gz'))
    pfw_files = glob.glob(os.path.join(node_dir, '*.pfw'))
    files = gz_files + pfw_files
    
    compact=os.path.join(node_dir,'COMPACT')
    if os.path.isdir(compact):
        files += glob.glob(os.path.join(compact, '*.pfw.gz'))
        files += glob.glob(os.path.join(compact, '*.pfw'))
        
    for p in sorted(files):
        if p.endswith('.pfw.gz'):
            # For gzipped files
            with gzip.open(p, 'rt') as f:
                for raw in f:
                    line = raw.strip().rstrip(',')
                    if not line or line in ('[',']'): continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    records.append(obj)
        else:
            # For non-gzipped .pfw files
            with open(p, 'r') as f:
                for raw in f:
                    line = raw.strip().rstrip(',')
                    if not line or line in ('[',']'): continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    records.append(obj)
                    
    if not records: return None
    
    # Create a pandas DataFrame directly, as Dask might not be necessary for smaller datasets
    df = pd.json_normalize(records)
    df['name'] = df.get('name', '').astype(str)
    df['dur'] = pd.to_numeric(df.get('dur', 0), errors='coerce').fillna(0.0)
    df['pid'] = pd.to_numeric(df.get('pid', 0), errors='coerce').fillna(0).astype(int)
    df['tid'] = pd.to_numeric(df.get('tid', 0), errors='coerce').fillna(0).astype(int)
    
    return df[['name','dur','pid','tid']]


def summarize_groups(df):
    df = df[df['dur']>0].copy()
    df['group'] = df['name'].apply(assign_group)
    
    # Convert to Dask only for larger dataframes
    if len(df) > 10000:  # Threshold for using Dask
        ddf = dd.from_pandas(df, npartitions=max(1, len(df) // 10000))
        total = ddf['dur'].sum().compute()
        
        # Use Dask aggregation
        agg_df = ddf.groupby('group')['dur'].agg(
            Total_Time='sum',
            Num_Instances='count',
            Average='mean',
            Min='min',
            Max='max',
            StdDev='std'
        ).reset_index().compute()
    else:
        total = df['dur'].sum()
        # Use pandas for smaller dataframes
        agg_df = df.groupby('group')['dur'].agg(
            Total_Time='sum',
            Num_Instances='count',
            Average='mean',
            Min='min',
            Max='max',
            StdDev='std'
        ).reset_index()
    
    agg_df['% Total Time'] = (100 * agg_df['Total_Time'] / total).round(3)
    for c in ['Total_Time','Average','Min','Max','StdDev']:
        agg_df[c] = agg_df[c].round(3)
    
    return agg_df.sort_values('% Total Time', ascending=False)[['group','% Total Time','Total_Time','Num_Instances','Average','Min','Max','StdDev']]


def summarize_events(df):
    df = df[df['dur']>0]
    
    # Convert to Dask only for larger dataframes
    if len(df) > 10000:  # Threshold for using Dask
        ddf = dd.from_pandas(df, npartitions=max(1, len(df) // 10000))
        total = ddf['dur'].sum().compute()
        
        # Use Dask aggregation
        agg_df = ddf.groupby('name')['dur'].agg(
            Total_Time='sum',
            Num_Instances='count',
            Average='mean',
            Min='min',
            Max='max',
            StdDev='std'
        ).reset_index().compute()
    else:
        total = df['dur'].sum()
        # Use pandas for smaller dataframes
        agg_df = df.groupby('name')['dur'].agg(
            Total_Time='sum',
            Num_Instances='count',
            Average='mean',
            Min='min',
            Max='max',
            StdDev='std'
        ).reset_index()
    
    agg_df['% Total Time'] = (100 * agg_df['Total_Time'] / total).round(3)
    for c in ['Total_Time','Average','Min','Max','StdDev']:
        agg_df[c] = agg_df[c].round(3)
    
    return agg_df.sort_values('% Total Time', ascending=False)[['name','% Total Time','Total_Time','Num_Instances','Average','Min','Max','StdDev']]


def summarize_process_groups(df):
    df = df[df['dur']>0].copy()
    df['group'] = df['name'].apply(assign_group)
    
    # Convert to Dask only for larger dataframes
    if len(df) > 10000:  # Threshold for using Dask
        ddf = dd.from_pandas(df, npartitions=max(1, len(df) // 10000))
        
        # Use Dask aggregation
        agg_df = ddf.groupby(['pid','group'])['dur'].agg(
            Total_Time='sum',
            Num_Instances='count',
            Average='mean',
            Min='min',
            Max='max',
            StdDev='std'
        ).reset_index().compute()
        
        # Calculate pid_total using Dask
        pct_df = ddf.groupby('pid')['dur'].sum().reset_index().rename(columns={'dur': 'pid_total'}).compute()
    else:
        # Use pandas for smaller dataframes
        agg_df = df.groupby(['pid','group'])['dur'].agg(
            Total_Time='sum',
            Num_Instances='count',
            Average='mean',
            Min='min',
            Max='max',
            StdDev='std'
        ).reset_index()
        
        # Calculate pid_total using pandas
        pct_df = df.groupby('pid')['dur'].sum().reset_index().rename(columns={'dur': 'pid_total'})
    
    # Merge results
    agg_df = agg_df.merge(pct_df, on='pid')
    agg_df['% Total Time'] = (100 * agg_df['Total_Time'] / agg_df['pid_total']).round(3)
    
    for c in ['Total_Time','Average','Min','Max','StdDev']:
        agg_df[c] = agg_df[c].round(3)
    
    return agg_df.sort_values(['pid','% Total Time'], ascending=[True,False])[['pid','group','% Total Time','Total_Time','Num_Instances','Average','Min','Max','StdDev']]


def summarize_thread_groups(df):
    df = df[df['dur']>0].copy()
    df['group'] = df['name'].apply(assign_group)
    
    # Convert to Dask only for larger dataframes
    if len(df) > 10000:  # Threshold for using Dask
        ddf = dd.from_pandas(df, npartitions=max(1, len(df) // 10000))
        
        # Use Dask aggregation
        agg_df = ddf.groupby(['tid','group'])['dur'].agg(
            Total_Time='sum',
            Num_Instances='count',
            Average='mean',
            Min='min',
            Max='max',
            StdDev='std'
        ).reset_index().compute()
        
        # Calculate tid_total using Dask
        pct_df = ddf.groupby('tid')['dur'].sum().reset_index().rename(columns={'dur': 'tid_total'}).compute()
    else:
        # Use pandas for smaller dataframes
        agg_df = df.groupby(['tid','group'])['dur'].agg(
            Total_Time='sum',
            Num_Instances='count',
            Average='mean',
            Min='min',
            Max='max',
            StdDev='std'
        ).reset_index()
        
        # Calculate tid_total using pandas
        pct_df = df.groupby('tid')['dur'].sum().reset_index().rename(columns={'dur': 'tid_total'})
    
    # Merge results
    agg_df = agg_df.merge(pct_df, on='tid')
    agg_df['% Total Time'] = (100 * agg_df['Total_Time'] / agg_df['tid_total']).round(3)
    
    for c in ['Total_Time','Average','Min','Max','StdDev']:
        agg_df[c] = agg_df[c].round(3)
    
    return agg_df.sort_values(['tid','% Total Time'], ascending=[True,False])[['tid','group','% Total Time','Total_Time','Num_Instances','Average','Min','Max','StdDev']]


def build_group_map(df):
    # Use pandas directly for better compatibility
    unique_names = df['name'].unique()
    
    gm = defaultdict(set)
    for nm in unique_names:
        gm[assign_group(nm)].add(nm)
    
    return {g: sorted(list(ev)) for g, ev in gm.items()}


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


def highlight_across_nodes(per_node, key_col, metrics):
    nodes = list(per_node.keys())
    
    # Extract unique keys from all DataFrames
    keys_set = set()
    for df in per_node.values():
        keys_set.update(df[key_col].tolist())
    keys = sorted(keys_set)
    
    for key in keys:
        print(f"\n>>> {key}")
        
        # Find max holder for each metric using improved approach
        max_holder = {}
        for m in metrics:
            max_val = float('-inf')
            max_node = None
            for n in nodes:
                row = per_node[n]
                if key in row[key_col].values:
                    val = row.set_index(key_col).get(m, 0).get(key, 0)
                    if val > max_val:
                        max_val = val
                        max_node = n
            max_holder[m] = max_node
        
        hdr = "    node" + "".join(m.rjust(12) for m in metrics)
        print(hdr)
        
        for n in nodes:
            row = per_node[n]
            if key in row[key_col].values:
                vals = {m: row.set_index(key_col).get(m, 0).get(key, 0) for m in metrics}
                line = "    " + n.ljust(11)
                for m in metrics:
                    col = Fore.RED if n == max_holder[m] else Fore.GREEN
                    line += col + f"{vals[m]:12.3f}" + Style.RESET_ALL
                print(line)
            else:
                # Print a placeholder for nodes that don't have this key
                line = "    " + n.ljust(11) + "".join(" " * 12 for _ in metrics)
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
    p.add_argument('--aggregate',action='store_true',help='Highlight the node with the max of each metric across nodes')
    p.add_argument('--level', type=int, default=0, help='Show additional levels of detail (1 or 2) in the tree view')
    args=p.parse_args()
    base=args.directory
    if not os.path.isdir(base): p.error(f"{base} not a dir")
    
    # Check for both .pfw.gz and .pfw files
    has_pfw_files = glob.glob(os.path.join(base,'*.pfw.gz')) or glob.glob(os.path.join(base,'*.pfw'))
    if has_pfw_files: 
        nodes=[base]
    else: 
        nodes=[os.path.join(base,d) for d in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base,d))]
    
    raw={}
    for nd in nodes:
        df=load_node_df(nd)
        if df is None: continue
        nm=os.path.basename(nd.rstrip(os.sep))
        raw[nm if nm.lower()!='compact' else os.path.basename(os.path.dirname(nd))]=df
    
    if not raw:
        print("No traces"); return
        
    # Use pandas concat for better compatibility
    combined_df = pd.concat(raw.values(), ignore_index=True)
    group_map = build_group_map(combined_df)
        
    mode=('events' if args.events else 'process' if args.process else 'thread' if args.thread else 'node')
    for n, df in raw.items():
        # Choose behavior for process + aggregate: ascii tree with avg only
        if mode == 'process' and args.aggregate:
            # Aggregate per-process: ascii tree by average, highlight group-wise max across all processes
            print(f"\n===== Summary for {n} (aggregate per-process) =====\n")
            proc_df = summarize_process_groups(df)
            
            # For level > 0, prepare event-level data for each process
            event_level_data = {}
            event_max_by_name = {}  # To store max avg time for each event across all processes
            
            if args.level > 0:
                # First pass: collect all event data by process
                for pid, sub_df in df.groupby('pid'):
                    event_df = summarize_events(sub_df)
                    event_level_data[pid] = event_df
                
                # Second pass: compute max avg time for each event across all processes
                for event_name in df['name'].unique():
                    max_avg = 0
                    max_pid = None
                    for pid, event_df in event_level_data.items():
                        event_rows = event_df[event_df['name'] == event_name]
                        if not event_rows.empty:
                            avg_time = event_rows.iloc[0]['Average']
                            if avg_time > max_avg:
                                max_avg = avg_time
                                max_pid = pid
                    if max_pid is not None:
                        event_max_by_name[event_name] = (max_pid, max_avg)
            
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
                    
                    # If level > 0, show individual events for this group
                    if args.level > 0 and pid in event_level_data:
                        # Get events for this group
                        events_in_group = []
                        for nm in group_map.get(grp, []):
                            # Find this event in the event data for this process
                            event_rows = event_level_data[pid][event_level_data[pid]['name'] == nm]
                            if not event_rows.empty:
                                events_in_group.append((nm, event_rows.iloc[0]['Average'], 
                                                       event_rows.iloc[0]['Total_Time'],
                                                       event_rows.iloc[0]['% Total Time'],
                                                       event_rows.iloc[0]))
                        
                        # Sort events by average time (descending)
                        events_in_group.sort(key=lambda x: x[1], reverse=True)
                        
                        # Display top events based on level (level 1 = top 3, level 2 = all)
                        if args.level == 1:
                            events_to_show = events_in_group[:min(3, len(events_in_group))]
                        else:
                            events_to_show = events_in_group
                        
                        for event_name, event_avg, event_total, event_pct, event_row in events_to_show:
                            # Print each event with indentation
                            print(f"    |")
                            event_line = f"    {prefix} {event_name} (avg: {event_avg:.6e}, total: {event_total:.6e}, %: {event_pct:.2f})"
                            
                            # Highlight in red if this process holds the max for this event
                            is_max = event_max_by_name.get(event_name, (None, 0))[0] == pid
                            if is_max:
                                print(Fore.RED + event_line + Style.RESET_ALL)
                            else:
                                print(event_line)
                            
                            # For level 2, add more detailed breakdown
                            if args.level >= 2:
                                # For level 2, we need to add more detail about each event
                                # If we have detailed metrics like subevent breakdowns, we could show them here
                                # For now, let's show additional stats from the event row
                                print(f"        |")
                                subdetail_line = f"        {prefix} Instances: {event_row['Num_Instances']}, Min: {event_row['Min']:.6e}, Max: {event_row['Max']:.6e}, StdDev: {event_row['StdDev']:.6e}"
                                print(subdetail_line)
                
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

