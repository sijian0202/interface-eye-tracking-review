"""Recalculate eye-tracking exports without writing source identities.

Usage:
  python tools/reanalyze.py --archive /local/raw.zip --public ./output-public \
      --private /separate/local-work --permutations 99999

The raw archive is read in memory. Only aggregates enter --public.
An identity key is never written. Coded per-record data stay in --private.
"""
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import re
import struct
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy import stats
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def zip_name(info):
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode('cp437').decode('gbk')
    except UnicodeError:
        return info.filename


def decode(data):
    if data.startswith((b'\xff\xfe', b'\xfe\xff')):
        return data.decode('utf-16')
    for enc in ('utf-8-sig', 'gb18030'):
        try:
            return data.decode(enc)
        except UnicodeError:
            continue
    raise ValueError('Unrecognized text encoding')


def number(value):
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (ValueError, TypeError):
        return None


def polygon_contains(x, y, polygon):
    # Inclusive boundary, ray crossing; no AOI geometry is inferred from images.
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        x1, y1 = a
        x2, y2 = b
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-11 and min(x1, x2)-1e-11 <= x <= max(x1, x2)+1e-11 and min(y1, y2)-1e-11 <= y <= max(y1, y2)+1e-11:
            return True
        if (y1 > y) != (y2 > y) and x < (x2-x1)*(y-y1)/(y2-y1)+x1:
            inside = not inside
    return inside


def area(poly):
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(poly,poly[1:]+poly[:1]))) / 2


def parse_rows(text):
    lines = text.splitlines()
    start = next(i for i,line in enumerate(lines) if line.startswith('Subject,Recording,'))
    return [r for r in csv.DictReader(lines[start:]) if r.get('Subject')]


def describe(values):
    vals = np.asarray([v for v in values if v is not None and math.isfinite(v)], float)
    if not len(vals):
        return {'n':0,'mean':None,'sd':None,'median':None,'q1':None,'q3':None}
    return {'n':len(vals), 'mean':float(np.mean(vals)),
            'sd':float(np.std(vals,ddof=1)) if len(vals)>1 else None,
            'median':float(np.median(vals)), 'q1':float(np.quantile(vals,.25)),
            'q3':float(np.quantile(vals,.75))}


def holm(pvalues):
    order = np.argsort(pvalues)
    result = np.empty(len(pvalues)); previous = 0.0
    for position,index in enumerate(order):
        previous = max(previous, (len(pvalues)-position)*pvalues[index])
        result[index] = min(previous,1.0)
    return result.tolist()


def friedman_permutation(matrix, repetitions, seed):
    n,k=matrix.shape
    ranks=stats.rankdata(matrix,axis=1,method='average')
    tie_sum=sum(sum(t**3-t for t in Counter(row).values()) for row in matrix)
    correction=1-tie_sum/(n*(k**3-k))
    if correction <= 0:
        return 0.0,1.0,0.0
    def qvalue(sums):
        return (12/(n*k*(k+1))*np.sum(sums*sums,axis=-1)-3*n*(k+1))/correction
    observed=max(0.0,float(qvalue(ranks.sum(axis=0))))
    rng=np.random.default_rng(seed); exceed=0
    for offset in range(0,repetitions,4096):
        b=min(4096,repetitions-offset)
        orders=np.argsort(rng.random((b,n,k)),axis=2)
        shuffled=np.take_along_axis(np.broadcast_to(ranks,(b,n,k)),orders,axis=2)
        null=qvalue(shuffled.sum(axis=1))
        exceed+=int(np.count_nonzero(null>=observed-1e-10))
    return observed,(exceed+1)/(repetitions+1),observed/(n*(k-1))


def write_csv(path, records):
    if not records:
        path.write_text('',encoding='utf-8')
        return
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--public',type=Path,required=True)
    parser.add_argument('--private',type=Path,required=True)
    parser.add_argument('--permutations',type=int,default=99999)
    args=parser.parse_args()
    pub=args.public.resolve(); private=args.private.resolve()
    if pub==private or pub in private.parents or private in pub.parents:
        raise SystemExit('Public and private output folders must be disjoint.')
    for folder in [pub/'data',pub/'figures',private]:folder.mkdir(parents=True,exist_ok=True)

    identity_tokens=set(); scene_specs={}; stimuli=[]; trial_time={}; tracks={}; coverage={}
    questions=[]; question_info=[]; root_csv_count=0
    with ZipFile(args.archive) as archive:
        items={zip_name(i):i for i in archive.infolist()}
        prefixes=sorted(n[:-len('Stimulations/SceneType.xml')] for n in items
                        if n.endswith('Stimulations/SceneType.xml') and '广告3/' in n)
        if len(prefixes)!=1:
            raise SystemExit('Expected one analysis project definition.')
        prefix=prefixes[0]
        scene_root=ET.fromstring(archive.read(items[prefix+'Stimulations/SceneType.xml']))
        for group in scene_root:
            if group.attrib.get('Type')!='Image':continue
            for scene in group:
                polygons=[]
                for node in scene.findall('AOIS/*'):
                    poly=[]
                    for child in node:
                        if child.tag.startswith('Point'):
                            pair=[struct.unpack('<d',struct.pack('<q',int(x)))[0] for x in child.text.split(',')]
                            if len(pair)!=2 or not all(math.isfinite(x) for x in pair):
                                raise ValueError('Invalid AOI coordinate encoding')
                            poly.append(pair)
                    if len(poly)>=3:
                        polygons.append({'name':node.findtext('Text'),'points':poly})
                scene_specs[scene.findtext('name')]={'typeid':scene.findtext('id'),'polygons':polygons}

        media=ET.fromstring(archive.read(items[prefix+'Stimulations/mediaType.xml']))
        current_instruction=''
        for block in media:
            kind=block.attrib.get('Type')
            if kind=='Text':current_instruction=block.findtext('instructionContent','')
            if kind=='Image':
                stimuli.append({'scene':block.findtext('elementName'),
                    'instruction':current_instruction,'manual_advance':block.findtext('manualPlay'),
                    'configured_seconds':block.findtext('viewTimeSec')})
            if kind=='Question':
                questions.append({'question':block.findtext('QuestionsName','').rstrip(','),
                                  'key':block.findtext('variableKey','').rstrip(',')})

        # Only info fields are used for answer summaries. Recording fields carry
        # conflicting/default values and must not be counted again.
        subj=ET.fromstring(archive.read(items[prefix+'Subjects/subjects.xml']))
        for sub in subj.findall('subject'):
            info=sub.find('info')
            if info is None:continue
            for leaf in info.iter():
                if leaf.tag == 'subjectName' and leaf.text:
                    identity_tokens.add(leaf.text.strip())
            children={c.tag:c for c in info}
            response={}
            for tag,node in children.items():
                if re.fullmatch(r'keys\d+',tag) and node.text in {'2','3','4','q5'}:
                    v=children.get('values'+tag[4:])
                    if v is not None:response[node.text]=(v.text or '').strip()
            question_info.append(response)

        for filename,entry in items.items():
            if '/' not in filename and filename.endswith('.csv'):
                root_csv_count+=1
                text=decode(archive.read(entry))
                for line in text.splitlines():
                    if line.startswith(('Subjects:','Recordings:')):
                        identity_tokens.update(x.strip() for x in line.split(':',1)[1].split(',') if x.strip())
                if filename.endswith('_GazeTrack.csv'):
                    scene=next((line.split(':',1)[1].strip() for line in text.splitlines()
                                if line.startswith('Scene Name:')),None)
                    if scene in scene_specs:
                        if scene in tracks:raise ValueError('Duplicate exported scene; review source selection.')
                        rows=parse_rows(text)
                        identity_tokens.update(r['Subject'] for r in rows)
                        identity_tokens.update(r['Recording'] for r in rows if r.get('Recording'))
                        tracks[scene]=rows
                if filename.endswith('_GazeAOI.csv'):
                    scene=filename[:-len('_GazeAOI.csv')]
                    lines=text.splitlines()
                    idx=next((i for i,line in enumerate(lines) if line.startswith('AOIName,')),None)
                    if idx is not None:
                        coverage[scene]={r['AOIName']:number(r['SubjectPercent'].strip('%'))
                                         for r in csv.DictReader(lines[idx:]) if r.get('AOIName')}
            if filename.startswith(prefix+'Recordings/') and filename.endswith('/Scenes/scene.xml'):
                recording=filename[len(prefix+'Recordings/'):].split('/')[0]
                identity_tokens.add(recording)
                for sc in ET.fromstring(archive.read(entry)).findall('Scene'):
                    delta=int(sc.findtext('endTime'))-int(sc.findtext('startTime'))
                    trial_time[(recording,sc.findtext('typeid'))]=delta/10_000_000

    all_people=sorted({r['Subject'] for rows in tracks.values() for r in rows})
    codes={person:f'P{index+1:02}' for index,person in enumerate(all_people)}
    by_scene={}; private_rows=[]; summary=[]; qa=[]; sensitivity=[]
    auto_durations=[]
    metrics=['fixation_time_s','fixation_count','aoi_time_s','aoi_count','aoi_share',
             'first_aoi_start_s','display_duration_s']
    for spec in stimuli:
        scene=spec['scene']
        if scene not in tracks:continue
        source=tracks[scene];groups=defaultdict(list)
        for row in source:groups[row['Subject']].append(row)
        records={}
        polygons=scene_specs[scene]['polygons']
        for person,rows in groups.items():
            valid=[];bad=0;dropped=0
            for row in rows:
                duration=number(row.get('Duration(s)')); start=number(row.get('StartTime(s)'))
                x=number(row.get('X(%)')); y=number(row.get('Y(%)'))
                if duration is None or duration<0:bad+=1;continue
                if duration<.05:dropped+=1;continue
                valid.append({'d':duration,'t':start,'x':x,'y':y,
                              'hit':bool(polygons and x is not None and y is not None and
                                         any(polygon_contains(x,y,p['points']) for p in polygons))})
            total=sum(r['d'] for r in valid)
            hit=[r for r in valid if r['hit']]
            target=sum(r['d'] for r in hit) if polygons else None
            latency=min((r['t'] for r in hit if r['t'] is not None),default=None)
            recording=rows[0].get('Recording',person)
            duration=trial_time.get((recording,scene_specs[scene]['typeid']))
            if spec['manual_advance']=='False' and duration is not None:auto_durations.append(duration)
            item={'participant':codes[person],'scene':scene,'fixation_time_s':total,
                  'fixation_count':len(valid),'aoi_time_s':target,
                  'aoi_count':len(hit) if polygons else None,
                  'aoi_share':target/total if polygons and total>0 else None,
                  'first_aoi_start_s':latency,'display_duration_s':duration,
                  'dropped_short_rows':dropped,'invalid_duration_rows':bad,
                  'unlocated_rows':sum(r['x'] is None or r['y'] is None for r in valid)}
            records[codes[person]]=item;private_rows.append(item)
            for threshold in [0.0,0.1]:
                candidates=[row for row in rows if number(row.get('Duration(s)')) is not None and float(row['Duration(s)'])>=threshold]
                sensitivity.append({'participant':codes[person],'scene':scene,'threshold_s':threshold,
                                    'fixation_time_s':sum(float(r['Duration(s)']) for r in candidates),
                                    'fixation_count':len(candidates)})
        by_scene[scene]=records
        recs=list(records.values())
        result={'scene':scene,'participants':len(recs),'aois':len(polygons),
                'aoi_coverage_count':sum(r['aoi_count']>0 for r in recs) if polygons else None,
                'aoi_area_fraction':sum(area(p['points']) for p in polygons) if polygons else None}
        for metric in metrics:
            for key,val in describe(r[metric] for r in recs).items():result[metric+'_'+key]=val
        summary.append(result)
        # Validate each named AOI against the software's subject coverage export.
        for poly in polygons:
            for threshold in [.05]:
                hit_people=set()
                for person,rows in groups.items():
                    for row in rows:
                        d=number(row.get('Duration(s)'));x=number(row.get('X(%)'));y=number(row.get('Y(%)'))
                        if d is not None and d>=threshold and x is not None and y is not None and polygon_contains(x,y,poly['points']):
                            hit_people.add(person);break
                derived=100*len(hit_people)/len(groups)
                expected=coverage.get(scene,{}).get(poly['name'])
                qa.append({'scene':scene,'aoi':poly['name'],'derived_subject_percent':round(derived,2),
                           'source_subject_percent':expected,
                           'matches':expected is not None and abs(derived-expected)<=.011})

    members=['会员1','会员2','会员3','会员4']
    common=sorted(set.intersection(*(set(by_scene[s]) for s in members)))
    if len(common)!=13:raise ValueError('Unexpected complete paired sample count; review before inference.')
    member_qa=[r for r in qa if r['scene'] in members]
    if not all(r['matches'] for r in member_qa):raise ValueError('Member AOI coverage validation failed.')
    test_metrics=['aoi_time_s','aoi_count','aoi_share']
    inference=[]
    for i,metric in enumerate(test_metrics):
        matrix=np.array([[by_scene[s][person][metric] for s in members] for person in common],float)
        statistic,p,w=friedman_permutation(matrix,args.permutations,20261017+i)
        # A second implementation of Q is a numerical cross-check, not the source
        # of p: asymptotic chi-square p is deliberately not used for four scenes.
        reference=float(stats.friedmanchisquare(*matrix.T).statistic)
        if not math.isclose(statistic,reference,abs_tol=1e-8):raise ValueError('Friedman statistic verification failed')
        inference.append({'metric':metric,'n':len(common),'scenes':4,'Q':statistic,
                          'permutation_p':p,'permutations':args.permutations,'kendall_W':w})
    for rec,adjusted in zip(inference,holm([r['permutation_p'] for r in inference])):
        rec['holm_p']=adjusted
    pairs=[]
    if inference[0]['holm_p']<.05:
        for a,b in itertools.combinations(members,2):
            diff=np.round(np.array([by_scene[a][pid]['aoi_time_s']-by_scene[b][pid]['aoi_time_s'] for pid in common]),9)
            if np.all(diff==0):statistic,p=0.0,1.0
            else:
                test=stats.wilcoxon(diff,zero_method='wilcox',alternative='two-sided',
                                    method=stats.PermutationMethod(n_resamples=np.inf))
                statistic,p=float(test.statistic),float(test.pvalue)
            pairs.append({'scene_a':a,'scene_b':b,'n_pairs':len(common),
                          'mean_paired_difference_s':float(diff.mean()),
                          'median_paired_difference_s':float(np.median(diff)),
                          'wilcoxon_W':statistic,'exact_signflip_p':p})
        for r,p in zip(pairs,holm([r['exact_signflip_p'] for r in pairs])):r['holm_p']=p

    qsummary=[]
    for question in questions:
        vals=[d[question['key']] for d in question_info if d.get(question['key'])]
        # Only anonymous category totals, never per-person answers, are public.
        for answer,count in sorted(Counter(vals).items()):
            qsummary.append({'question_key':question['key'],'question':question['question'],
                             'answer':answer,'count':count,'available_info_entries':len(vals)})

    sensitivity_summary=[]
    for scene in by_scene:
        for threshold in [0.0,.05,.1]:
            if threshold==.05:
                recs=list(by_scene[scene].values())
            else:recs=[r for r in sensitivity if r['scene']==scene and r['threshold_s']==threshold]
            sensitivity_summary.append({'scene':scene,'threshold_s':threshold,'n':len(recs),
                'mean_fixation_time_s':float(np.mean([r['fixation_time_s'] for r in recs])),
                'mean_fixation_count':float(np.mean([r['fixation_count'] for r in recs]))})

    write_csv(private/'coded_scene_metrics.csv',private_rows)
    write_csv(private/'coded_threshold_sensitivity.csv',sensitivity)
    write_csv(pub/'data/scene_summary.csv',summary)
    write_csv(pub/'data/member_tests.csv',inference)
    write_csv(pub/'data/member_posthoc.csv',pairs)
    write_csv(pub/'data/question_counts.csv',qsummary)
    write_csv(pub/'data/threshold_sensitivity.csv',sensitivity_summary)
    write_csv(pub/'data/aoi_coverage_checks.csv',qa)
    write_csv(pub/'data/task_catalog.csv',stimuli)

    results={'analysis_date':'2026-10-17','input_sha256':hashlib.sha256(args.archive.read_bytes()).hexdigest(),
             'root_csv_count':root_csv_count,'scene_count':len(by_scene),'defined_image_scenes':len(stimuli),
             'participants_in_exports':len(all_people),'member_complete_paired_n':len(common),
             'member_aoi_checks_all_passed':all(r['matches'] for r in member_qa),
             'all_aoi_checks_passed':all(r['matches'] for r in qa),
             'aoi_check_mismatches':[r for r in qa if not r['matches']],
             'auto_scene_count':len(auto_durations),
             'auto_scene_time_range_s':[min(auto_durations),max(auto_durations)] if auto_durations else None,
             'identity_key_written':False,'private_rows_published':False,
             'software':{'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
             'member_tests':inference,'member_posthoc':pairs,'scene_summary':summary,'question_counts':qsummary}
    (pub/'data/results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')

    # Generated charts contain aggregates/distributions only; never IDs or
    # raw stimulus screenshots, which may expose account information.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for ax,metric,title,ylabel in [(axes[0],'aoi_time_s','Target-region fixation time','Seconds'),
                                 (axes[1],'aoi_share','Target-region share of fixation time','Fraction of scene fixation time')]:
        vals=[[by_scene[s][pid][metric] for pid in common] for s in members]
        ax.boxplot(vals,tick_labels=['M1','M2','M3','M4'],showmeans=True,showfliers=False,
                   meanprops={'marker':'D','markerfacecolor':'#126b78','markeredgecolor':'#126b78','markersize':5},
                   medianprops={'color':'#c86736','linewidth':2})
        ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2)
        if metric=='aoi_share':ax.set_ylim(0,1.03)
    fig.suptitle('Four member-area scenes | same 13 participants',fontsize=14)
    fig.savefig(pub/'figures/member_overview.png',dpi=180,metadata={'Software':'Matplotlib; aggregate reanalysis'})
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.2),layout='constrained')
    english={'位置':'P1','位置2':'P2','位置3':'P3','位置4':'P4','信息流':'F1','信息流1':'F2','信息流2':'F3',
             '会员1':'M1','会员2':'M2','会员3':'M3','会员4':'M4','任务类型':'T1','任务类型-Copy':'T2',
             '1 (2)':'E1','2 (2)':'E2','3 (2)':'E3'}
    labels=[english.get(r['scene'],r['scene']) for r in summary]
    means=[r['fixation_time_s_mean'] for r in summary]
    colors=[{'P':'#6c86a3','F':'#ab7e55','M':'#247f82','T':'#8a73a4','E':'#869260'}[label[0]] for label in labels]
    ax.bar(labels,means,color=colors)
    ax.set_ylim(0,max(means)*1.20)
    ax.set_title('Exported scene fixation-time sums | descriptive only')
    ax.set_ylabel('Mean seconds per participant');ax.grid(axis='y',alpha=.2)
    ax.text(.01,.97,'Different tasks, content and exposure times; no causal ranking.',transform=ax.transAxes,va='top',fontsize=9)
    fig.savefig(pub/'figures/all_scenes.png',dpi=180,metadata={'Software':'Matplotlib; aggregate reanalysis'})
    plt.close(fig)

    # Scan every publishable file for any recovered identity token. Source
    # identity tokens and the mapping are never saved, even to the private folder.
    violations=[]
    for path in pub.rglob('*'):
        if not path.is_file():continue
        raw=path.read_bytes()
        for token in identity_tokens:
            if len(token)<2:continue
            needles=[token.encode('utf-8'),token.encode('utf-16le'),json.dumps(token,ensure_ascii=True)[1:-1].encode('ascii')]
            if any(needle in raw for needle in needles):violations.append(path.relative_to(pub).as_posix());break
    if violations:raise SystemExit('Identity scan failed in: '+','.join(sorted(set(violations))))
    privacy={'checked_files':sum(p.is_file() for p in pub.rglob('*')),
             'recovered_identity_tokens_checked':len(identity_tokens),'identity_matches':0,
             'raw_archive_published':False,'participant_rows_published':False,
             'identity_mapping_written':False,'stimulus_screenshots_published':False}
    (pub/'data/privacy_check.json').write_text(json.dumps(privacy,indent=2),encoding='utf-8')
    print(json.dumps({'scenes':len(summary),'member_n':len(common),'aoi_checks_member_passed':True,
                      'aoi_mismatch_scenes':[r['scene'] for r in qa if not r['matches']],
                      'member_tests':inference,'posthoc':pairs,'privacy':privacy},ensure_ascii=False))


if __name__=='__main__':
    main()
