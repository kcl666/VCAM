"""
Enhanced Combo Extraction Script（Coverage-Constrained Sampling）

Objective: Generate M combinations for each sample (1 main view + 11 candidates) (each combination contains k candidates),
And ensure that each candidate viewpoint appears at least min_cover times, and that there is diversity between combinat
ions (symmetric difference >= diversity_thresh).

Output structure（out_root）：
  out_root/<sample_id>/
    gt.obj
    meta.txt
    meta.json
    combo_01/
       000.png  # mian image
       001.png  # 1..k renumber
       ...
       camera.json

"""
import os
import json
import random
import shutil
import math
from PIL import Image
from typing import List, Dict, Tuple
from tqdm import tqdm

# -----------------------
# Helpers (camera parsing, angles, sampling)
# -----------------------

def load_camera_json(path: str):
    """
    支持：
      - list of dicts: [{ 'view_id': int, 'az':..., 'el':..., 'roll':... }, ...]
      - pose.json-like: { 'intrinsics':..., 'near_far':[...], 'c2ws': { 'name': matrix, ... }}
    返回统一 list of dicts: { 'orig_name', 'view_idx', 'az', 'el', 'roll', 'c2w'(opt) }
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entries = []
    if isinstance(data, list):
        for item in data:
            vid = item.get('view_id', None)
            try:
                vid_int = int(vid) if vid is not None else None
            except:
                vid_int = None
            fname = item.get('orig_name', None) or (f"{vid_int:03d}.png" if vid_int is not None else None)
            az = float(item.get('az', 0.0))
            el = float(item.get('el', 0.0))
            roll = float(item.get('roll', 0.0))
            entries.append({'orig_name': fname, 'view_idx': vid_int, 'az': az, 'el': el, 'roll': roll})
        return entries

    if isinstance(data, dict) and 'c2ws' in data:
        c2ws = data['c2ws']
        keys = sorted(list(c2ws.keys()))
        for k in keys:
            base = os.path.basename(k)
            name = base
            try:
                idx = int(os.path.splitext(base)[0])
            except:
                idx = None
            entries.append({'orig_name': name, 'view_idx': idx if idx is not None else -1, 'c2w': c2ws[k]})
        return entries

    raise ValueError(f"Unsupported camera.json format at {path}")


def angular_distance(a1, e1, a2, e2):
    a1r, e1r = math.radians(a1), math.radians(e1)
    a2r, e2r = math.radians(a2), math.radians(e2)
    x1 = math.cos(e1r) * math.cos(a1r)
    y1 = math.cos(e1r) * math.sin(a1r)
    z1 = math.sin(e1r)
    x2 = math.cos(e2r) * math.cos(a2r)
    y2 = math.cos(e2r) * math.sin(a2r)
    z2 = math.sin(e2r)
    dot = x1*x2 + y1*y2 + z1*z2
    dot = max(-1.0, min(1.0, dot))
    angle = math.degrees(math.acos(dot))
    return angle


def farthest_sampling_by_pose(candidates: List[Dict], k: int, rng: random.Random) -> List[Dict]:
    # candidates must have 'az' and 'el'
    if len(candidates) <= k:
        return list(candidates)
    first = rng.choice(candidates)
    picked = [first]
    remaining = [c for c in candidates if c != first]
    while len(picked) < k:
        best = None
        best_min = -1
        for cand in remaining:
            min_d = min(angular_distance(cand['az'], cand['el'], p['az'], p['el']) for p in picked)
            if min_d > best_min:
                best_min = min_d; best = cand
        if best is None:
            best = remaining[0]
        picked.append(best)
        remaining.remove(best)
    return picked


def symmetric_difference_size(a: List[int], b: List[int]) -> int:
    return len(set(a).symmetric_difference(set(b)))

# -----------------------
# Main extraction per-sample
# -----------------------

def process_sample(orig_sample_dir: str, out_root: str, M: int=12, k: int=5, min_cover: int=3,
                   resize: Tuple[int,int]=(256,256), seed: int=42, diversity_thresh: int=2, max_attempts: int=500):
    rng = random.Random(seed)

    # find camera
    cam_path = None
    for cand in ('camera.json', 'pose.json'):
        p = os.path.join(orig_sample_dir, cand)
        if os.path.exists(p):
            cam_path = p; break
    if cam_path is None:
        return False, 'no_camera'

    objs = [f for f in os.listdir(orig_sample_dir) if f.lower().endswith('.obj')]
    if len(objs) == 0:
        return False, 'no_gt'
    gt_name = objs[0]

    try:
        cam_entries = load_camera_json(cam_path)
    except Exception as e:
        return False, 'bad_camera'

    img_files_all = [f for f in os.listdir(orig_sample_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    if len(img_files_all) == 0:
        return False, 'no_img'

    # map camera info
    cam_map = {e['orig_name']: e for e in cam_entries if e.get('orig_name')}

    # main: prefer '000.png', else choose lowest view_idx, else first sorted
    main_name = '000.png' if '000.png' in img_files_all else None
    if main_name is None:
        possible = [e for e in cam_entries if e.get('view_idx') is not None and e['view_idx']>=0]
        if possible:
            possible_sorted = sorted(possible, key=lambda x: x['view_idx'])
            main_name = possible_sorted[0]['orig_name']
        else:
            main_name = sorted(img_files_all)[0]

    candidates = [n for n in img_files_all if n != main_name]
    if len(candidates) < k:
        return False, 'too_few'

    # build candidate entries with az/el (fallback synthetic)
    cand_entries = []
    for c in candidates:
        if c in cam_map and 'az' in cam_map[c]:
            cand_entries.append({'orig_name': c, 'az': cam_map[c]['az'], 'el': cam_map[c]['el'], 'roll': cam_map[c].get('roll',0.0)})
        else:
            # synthetic az by file index
            try:
                idx = int(os.path.splitext(c)[0])
            except:
                idx = abs(hash(c)) % 360
            cand_entries.append({'orig_name': c, 'az': float((idx*30) % 360), 'el': 0.0, 'roll': 0.0})

    # prepare output dirs
    sample_id = os.path.basename(orig_sample_dir.rstrip('/\\'))
    out_sample_dir = os.path.join(out_root, sample_id)
    os.makedirs(out_sample_dir, exist_ok=True)
    # copy GT once
    src_gt = os.path.join(orig_sample_dir, gt_name)
    dst_gt = os.path.join(out_sample_dir, gt_name)
    if not os.path.exists(dst_gt):
        shutil.copy2(src_gt, dst_gt)

    # prepare sampling
    combos: List[List[str]] = []  # list of lists of orig_name of chosen aux views
    counts = {c['orig_name']: 0 for c in cand_entries}

    attempts = 0
    # Strategy: weighted sampling favoring low-count views until we reach M combos
    while len(combos) < M and attempts < max_attempts:
        attempts += 1
        # sampling weights inversely proportional to (1+count)
        names = [c['orig_name'] for c in cand_entries]
        weights = [1.0 / (1.0 + counts[n]) for n in names]
        # normalize
        s = sum(weights)
        if s == 0:
            probs = [1.0/len(weights)]*len(weights)
        else:
            probs = [w/s for w in weights]
        # sample k without replacement with probabilities
        chosen = []
        pool = names[:]
        pool_probs = probs[:]
        for _ in range(k):
            # normalize each step
            s2 = sum(pool_probs)
            if s2 <= 0:
                idx = rng.randrange(len(pool))
            else:
                r = rng.random()*s2
                acc = 0.0
                idx = 0
                while idx < len(pool):
                    acc += pool_probs[idx]
                    if r <= acc:
                        break
                    idx += 1
                if idx >= len(pool): idx = len(pool)-1
            chosen_name = pool.pop(idx)
            pool_probs.pop(idx)
            chosen.append(chosen_name)

        # enforce diversity vs existing combos
        chosen_idx = [int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else None for x in chosen]
        # compute symmetric diff size against existing combos (by view indices if possible)
        ok_div = True
        for ex in combos:
            # compare using names
            diff = symmetric_difference_size(chosen, ex)
            if diff < diversity_thresh:
                ok_div = False; break
        if not ok_div:
            continue

        # accept combo
        combos.append(chosen)
        for n in chosen:
            counts[n] += 1

    # After initial loop, ensure min_cover: if some views < min_cover, try to add combos to cover them
    all_view_names = [c['orig_name'] for c in cand_entries]
    low_views = [v for v, cnt in counts.items() if cnt < min_cover]
    extra_attempts = 0
    while low_views and extra_attempts < max_attempts:
        extra_attempts += 1
        # build combo that prioritizes low_views: include up to k from low_views then fill rest by farthest sampling
        pick = []
        need = k
        rng.shuffle(low_views)
        for v in low_views:
            if v not in pick:
                pick.append(v); need -= 1
                if need == 0: break
        # fill remaining using farthest sampling on remaining candidates
        remaining_pool = [c for c in cand_entries if c['orig_name'] not in pick]
        if remaining_pool and need > 0:
            # use farthest from those in pick or random if pick empty
            if pick:
                # compute an az/el list for pick center
                chosen_pose_objs = [next(ci for ci in cand_entries if ci['orig_name']==pn) for pn in pick]
                # greedy add
                while need > 0 and remaining_pool:
                    best = None; best_min = -1
                    for cand in remaining_pool:
                        min_d = min(angular_distance(cand['az'], cand['el'], p['az'], p['el']) for p in chosen_pose_objs)
                        if min_d > best_min:
                            best_min = min_d; best = cand
                    if best is None: break
                    pick.append(best['orig_name']); chosen_pose_objs.append(best); remaining_pool.remove(best); need -= 1
            else:
                # random fill
                avail = [c['orig_name'] for c in remaining_pool]
                rng.shuffle(avail)
                pick += avail[:need]
                need = 0

        # diversity check
        ok_div = True
        for ex in combos:
            if symmetric_difference_size(pick, ex) < diversity_thresh:
                ok_div = False; break
        if not ok_div:
            continue

        combos.append(pick)
        for n in pick:
            counts[n] += 1
        low_views = [v for v, cnt in counts.items() if cnt < min_cover]

        # limit combos size to a soft cap: allow up to M + (M//2)
        if len(combos) >= M + (M//2):
            break

    # If still low_views not satisfied, we will redistribute by greedy replacement: attempt to modify existing combos
    if any(cnt < min_cover for cnt in counts.values()):
        # try targeted augmentation by replacing one element in some combos
        for v_need, cnt in list(counts.items()):
            while counts[v_need] < min_cover:
                # find a combo that doesn't contain v_need and where replacement keeps diversity
                replaced = False
                for idx, ex in enumerate(combos):
                    # pick candidate in ex to replace that has highest current count (>min_cover)
                    replace_candidates = sorted(ex, key=lambda x: counts[x], reverse=True)
                    for rc in replace_candidates:
                        if counts[rc] > min_cover:
                            new_combo = [x for x in ex if x != rc] + [v_need]
                            # check diversity vs other combos
                            ok = True
                            for j, other in enumerate(combos):
                                if j == idx: continue
                                if symmetric_difference_size(new_combo, other) < diversity_thresh:
                                    ok = False; break
                            if not ok:
                                continue
                            # perform replacement
                            combos[idx] = new_combo
                            counts[v_need] += 1
                            counts[rc] -= 1
                            replaced = True
                            break
                    if replaced: break
                if not replaced:
                    break

    # trim/pad combos to exactly M by truncation or by additional random diverse picks
    if len(combos) > M:
        combos = combos[:M]
    while len(combos) < M:
        # add random farthest-sampled combo
        pick = farthest_sampling_by_pose(cand_entries, k, rng)
        pick_names = [p['orig_name'] for p in pick]
        ok_div = True
        for ex in combos:
            if symmetric_difference_size(pick_names, ex) < diversity_thresh:
                ok_div = False; break
        if ok_div:
            combos.append(pick_names)
        else:
            # fallback random
            all_names = [c['orig_name'] for c in cand_entries]
            random_choice = rng.sample(all_names, k)
            combos.append(random_choice)

    # finalize: write combos to disk with renaming 000..005 and camera.json per combo
    meta = {'source': os.path.abspath(orig_sample_dir), 'gt': gt_name, 'combos': []}
    for ci, combo in enumerate(combos, start=1):
        combo_name = f"combo_{ci:02d}"
        combo_dir = os.path.join(out_sample_dir, combo_name)
        os.makedirs(combo_dir, exist_ok=True)
        # write main
        src_main = os.path.join(orig_sample_dir, main_name)
        dst_main = os.path.join(combo_dir, f"{0:03d}.png")
        img = Image.open(src_main).convert('RGB')
        if resize:
            img = img.resize(resize, Image.Resampling.LANCZOS)
        img.save(dst_main)
        camera_list = []
        main_cam = cam_map.get(main_name, {'az':0.0, 'el':0.0, 'roll':0.0})
        camera_list.append({'view_id': 0, 'orig_name': main_name, 'az': main_cam.get('az', 0.0), 'el': main_cam.get('el', 0.0), 'roll': main_cam.get('roll', 0.0)})

        for j, orig in enumerate(combo, start=1):
            src = os.path.join(orig_sample_dir, orig)
            dst = os.path.join(combo_dir, f"{j:03d}.png")
            img = Image.open(src).convert('RGB')
            if resize:
                img = img.resize(resize, Image.Resampling.LANCZOS)
            img.save(dst)
            caminfo = cam_map.get(orig, {'az':0.0, 'el':0.0, 'roll':0.0})
            camera_list.append({'view_id': j, 'orig_name': orig, 'az': caminfo.get('az',0.0), 'el': caminfo.get('el',0.0), 'roll': caminfo.get('roll',0.0)})

        with open(os.path.join(combo_dir, 'camera.json'), 'w', encoding='utf-8') as f:
            json.dump(camera_list, f, indent=2, ensure_ascii=False)

        meta['combos'].append({'combo_name': combo_name, 'images': [f"{i:03d}.png" for i in range(0, k+1)], 'orig_images': [main_name] + combo, 'camera': camera_list})

    # write meta files
    meta_txt_path = os.path.join(out_sample_dir, 'meta.txt')
    with open(meta_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"source: {meta['source']}\n")
        f.write(f"gt: {meta['gt']}\n")
        f.write(f"n_combos: {len(meta['combos'])}\n")
        for c in meta['combos']:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    with open(os.path.join(out_sample_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return True, 'ok'


def find_samples_in_root(root_dir: str):
    samples = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        files_lower = [f.lower() for f in filenames]
        has_cam = any(x in files_lower for x in ('camera.json','pose.json'))
        has_obj = any(f.endswith('.obj') for f in filenames)
        imgs = [f for f in filenames if f.lower().endswith(('.png','.jpg','.jpeg'))]
        if has_cam and has_obj and len(imgs) >= 6:
            samples.append(dirpath)
    return samples

# -----------------------
# CLI
# -----------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--renders_root', type=str, required=True)
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--M', type=int, default=12)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--min_cover', type=int, default=3)
    parser.add_argument('--resize', type=int, nargs=2, default=(256,256))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--diversity_thresh', type=int, default=2)
    parser.add_argument('--max_attempts', type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    samples = find_samples_in_root(args.renders_root)
    print(f"Found {len(samples)} samples")
    skipped = []
    for s in tqdm(samples):
        ok, reason = process_sample(s, args.out_root, M=args.M, k=args.k, min_cover=args.min_cover,
                                    resize=tuple(args.resize), seed=args.seed, diversity_thresh=args.diversity_thresh,
                                    max_attempts=args.max_attempts)
        if not ok:
            skipped.append((s, reason))
    if skipped:
        print('Skipped:')
        for p, r in skipped:
            print(p, r)
    print('Done.')
