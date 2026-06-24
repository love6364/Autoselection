"""
Core engine for Diamond Auto-Selection.
Kept separate from app.py so the matching logic can be unit-tested
without needing a running Streamlit session.
"""
import difflib
import re

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Constants / defaults
# ----------------------------------------------------------------------

MASTER_ALIASES = {
    'shape': ['shape'],
    'from_size': ['from size', 'from', 'min size', 'size from', 'fromsize'],
    'to_size': ['to size', 'to', 'max size', 'size to', 'tosize'],
    'clarity': ['clarity'],
    'color': ['color', 'colour'],
    'grid': ['grid', 'required', 'requirement', 'max pcs', 'target', 'need'],
    'available': ['available', 'avl', 'in stock', 'stock', 'on hand'],
}

PARTY_ALIASES = {
    'shape': ['shape'],
    'color': ['color', 'colour', 'fancy color', 'fancycolor'],
    'clarity': ['clarity'],
    'carat': ['carat', 'weight', 'size', 'cts', 'ct.'],
    'cut': ['cut'],
    'polish': ['polish', 'pol'],
    'symmetry': ['symmetry', 'symm', 'sym'],
    'ratio': ['ratio'],
    'length': ['length', 'len'],
    'width': ['width', 'wid'],
    'measurement': ['measurement', 'measurements', 'meas'],
}

GRADE_SYNONYMS = {
    'IDEAL': 'ID', 'ID': 'ID',
    'EXCELLENT': 'EX', 'EX': 'EX',
    'VERY GOOD': 'VG', 'VERYGOOD': 'VG', 'VG': 'VG',
    'GOOD': 'GD', 'GD': 'GD',
    'FAIR': 'FR', 'FR': 'FR',
    'POOR': 'PR', 'PR': 'PR',
}
STANDARD_PASS_GRADES = {'ID', 'EX'}

SHAPE_ALIASES = {
    'ROUND BRILLIANT': 'ROUND', 'RBC': 'ROUND', 'RND': 'ROUND', 'RB': 'ROUND',
    'EMERALD CUT': 'EMERALD', 'EM': 'EMERALD',
    'OVAL CUT': 'OVAL', 'OV': 'OVAL', 'ELONGATED OVAL': 'OVAL', 'MOVAL': 'OVAL',
    'ASHOKA': 'CUSHION MODIFIED', 'CUSHION MODIFIED': 'CUSHION MODIFIED',
    'ELONGATED CUSHION': 'CUSHION MODIFIED', 'OLD MINE PEAR': 'CUSHION MODIFIED',
    'CUSHION SQUARE': 'CUSHION BRILLIANT', 'SQUARE CUSHION': 'CUSHION BRILLIANT',
    'CUSHION BRILLIANT': 'CUSHION BRILLIANT', 'CUSHION': 'CUSHION BRILLIANT',
    'RADIANT CUT': 'RADIANT', 'CRISSCUT RADIANT': 'RADIANT', 'CRISS CUT': 'RADIANT',
    'PRINCESS CUT': 'PRINCESS', 'PRIN': 'PRINCESS',
    'MARQUISE CUT': 'MARQUISE', 'DUTCH MARQUISE': 'MARQUISE',
    'PEAR SHAPE': 'PEAR', 'PEAR MODIFIED': 'PEAR', 'ELONGATED PEAR': 'PEAR',
    'HEART SHAPE': 'HEART',
    'ASSCHER CUT': 'ASSCHER',
}

DEFAULT_RATIO_STANDARDS = pd.DataFrame([
    {'Shape': 'ASSCHER', 'Min Ratio': 1.00, 'Max Ratio': 1.10},
    {'Shape': 'ROUND', 'Min Ratio': 1.15, 'Max Ratio': 1.50},
])

REASON_NO_GRID = 'No matching Shape/Color/Clarity/Size bucket in requirement grid'
REASON_SHAPE_UNMAPPED = 'Shape not mapped to a requirement-grid shape'
REASON_CARAT_BAD = 'Carat / Size value could not be read'
REASON_FULFILLED = 'Requirement already fulfilled for this Shape/Color/Clarity/Size bucket'


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def norm_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ''
    s = str(x).strip()
    if s.lower() == 'nan':
        return ''
    return s.upper()


def norm_clarity(x) -> str:
    return norm_str(x).replace(' ', '')


def norm_grade(x) -> str:
    s = norm_str(x)
    return GRADE_SYNONYMS.get(s, s)


def grade_passes_standard(x) -> bool:
    s = norm_grade(x)
    if s == '':
        return True  # blank treated as EX, per spec
    return s in STANDARD_PASS_GRADES


def parse_carat(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x) if not np.isnan(x) else np.nan
    m = re.search(r'\d+\.?\d*', str(x))
    return float(m.group()) if m else np.nan


def parse_ratio_from_measurement(x):
    if x is None:
        return np.nan
    nums = re.findall(r'\d+\.?\d*', str(x))
    if len(nums) >= 2:
        a, b = float(nums[0]), float(nums[1])
        if b == 0:
            return np.nan
        r = a / b
        return r if r >= 1 else (1 / r)
    return np.nan


def guess_column(columns, aliases):
    cols = {c: str(c).strip().lower() for c in columns}
    for alias in aliases:
        for c, cl in cols.items():
            if cl == alias:
                return c
    for alias in aliases:
        for c, cl in cols.items():
            if alias in cl:
                return c
    return None


def guess_header_row(raw_df: pd.DataFrame, max_scan: int = 20) -> int:
    keywords = {
        'shape', 'color', 'colour', 'clarity', 'carat', 'weight', 'size',
        'cut', 'polish', 'symmetry', 'symm', 'ratio', 'lab', 'stockno',
        'stoneno', 'stock id', 'id', 'status', 'measurement', 'measurements',
    }
    best_row, best_score = 0, -1
    for i in range(min(max_scan, len(raw_df))):
        row = raw_df.iloc[i]
        non_null = int(row.notna().sum())
        text_hits = sum(
            1 for v in row
            if isinstance(v, str) and v.strip().lower() in keywords
        )
        score = text_hits * 10 + non_null
        if score > best_score:
            best_score, best_row = score, i
    return best_row


def guess_shape_mapping(raw_shape, master_shapes):
    s = norm_str(raw_shape)
    if not s:
        return None
    if s in master_shapes:
        return s
    if s in SHAPE_ALIASES and SHAPE_ALIASES[s] in master_shapes:
        return SHAPE_ALIASES[s]
    for k, v in SHAPE_ALIASES.items():
        if k in s and v in master_shapes:
            return v
    for ms in master_shapes:
        if ms in s or s in ms:
            return ms
    match = difflib.get_close_matches(s, master_shapes, n=1, cutoff=0.55)
    return match[0] if match else None


# ----------------------------------------------------------------------
# Requirement grid (master file)
# ----------------------------------------------------------------------

def build_requirement_df(raw_df: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    df = pd.DataFrame()
    df['Shape'] = raw_df[colmap['shape']].apply(norm_str)
    df['From Size'] = pd.to_numeric(raw_df[colmap['from_size']], errors='coerce')
    df['To Size'] = pd.to_numeric(raw_df[colmap['to_size']], errors='coerce')
    df['Clarity'] = raw_df[colmap['clarity']].apply(norm_clarity)
    df['Color'] = raw_df[colmap['color']].apply(norm_str)
    df['Grid'] = pd.to_numeric(raw_df[colmap['grid']], errors='coerce').fillna(0)
    df['Available'] = pd.to_numeric(raw_df[colmap['available']], errors='coerce').fillna(0)
    df = df.dropna(subset=['From Size', 'To Size'])
    df['Need to Buy'] = (df['Grid'] - df['Available']).clip(lower=0).astype(int)
    df['Size Group'] = df['From Size'].map(lambda v: f"{v:.2f}") + '-' + df['To Size'].map(lambda v: f"{v:.2f}")
    df = df.reset_index(drop=True)
    df.insert(0, 'Bucket ID', df.index)
    df['Need Remaining'] = df['Need to Buy']
    return df


def build_bucket_index(req_df: pd.DataFrame) -> dict:
    idx = {}
    for _, row in req_df.iterrows():
        key = (row['Shape'], row['Clarity'], row['Color'])
        idx.setdefault(key, []).append((row['From Size'], row['To Size'], row['Bucket ID']))
    return idx


def find_bucket(shape, clarity, color, carat, bucket_index):
    if not shape or pd.isna(carat):
        return None
    candidates = bucket_index.get((shape, clarity, color))
    if not candidates:
        return None
    for lo, hi, bid in candidates:
        if lo <= carat <= hi:
            return bid
    return None


# ----------------------------------------------------------------------
# Party file matching
# ----------------------------------------------------------------------

def prepare_party_df(party_df: pd.DataFrame, mapping: dict, shape_map: dict,
                      modes: dict, ratio_standards: pd.DataFrame,
                      undefined_shape_allow: bool, master_shapes: list) -> pd.DataFrame:
    """Adds normalized helper columns (prefixed with _) used for matching."""
    df = party_df.copy()

    df['_Shape'] = df[mapping['shape']].apply(lambda v: shape_map.get(norm_str(v)))
    df['_Clarity'] = df[mapping['clarity']].apply(norm_clarity)
    df['_Color'] = df[mapping['color']].apply(norm_str)
    df['_Carat'] = df[mapping['carat']].apply(parse_carat)

    cps_mode = modes.get('cps_mode', 'Standard')
    for field in ['cut', 'polish', 'symmetry']:
        col = mapping.get(field)
        if col and cps_mode == 'Standard':
            df[f'_{field}_pass'] = df[col].apply(grade_passes_standard)
        else:
            df[f'_{field}_pass'] = True

    ratio_source = modes.get('ratio_source', 'none')
    if ratio_source == 'existing' and mapping.get('ratio'):
        df['_Ratio'] = pd.to_numeric(df[mapping['ratio']], errors='coerce')
    elif ratio_source == 'measurement' and mapping.get('measurement'):
        df['_Ratio'] = df[mapping['measurement']].apply(parse_ratio_from_measurement)
    elif ratio_source == 'length_width' and mapping.get('length') and mapping.get('width'):
        L = pd.to_numeric(df[mapping['length']], errors='coerce')
        W = pd.to_numeric(df[mapping['width']], errors='coerce')
        r = L / W
        df['_Ratio'] = r.where((r >= 1) | r.isna(), 1 / r)
    else:
        df['_Ratio'] = np.nan

    ratio_check_mode = modes.get('ratio_check_mode', 'Specific')
    if ratio_check_mode == 'Standard' and ratio_source != 'none':
        std_map = {row['Shape']: (row['Min Ratio'], row['Max Ratio'])
                   for _, row in ratio_standards.iterrows()}

        def _check(row):
            shp, r = row['_Shape'], row['_Ratio']
            if shp in std_map:
                lo, hi = std_map[shp]
                if pd.isna(r):
                    return False, 'Ratio missing/unparseable'
                if lo <= r <= hi:
                    return True, ''
                return False, f'Ratio {r:.3f} outside standard {lo}-{hi} for {shp}'
            return (True, '') if undefined_shape_allow else (False, f'No ratio standard defined for shape {shp}')

        res = df.apply(_check, axis=1)
        df['_ratio_pass'] = res.apply(lambda t: t[0])
        df['_ratio_reason'] = res.apply(lambda t: t[1])
    else:
        df['_ratio_pass'] = True
        df['_ratio_reason'] = ''

    bucket_index = mapping['_bucket_index']
    df['_BucketID'] = df.apply(
        lambda r: find_bucket(r['_Shape'], r['_Clarity'], r['_Color'], r['_Carat'], bucket_index),
        axis=1,
    )

    def _fail_reasons(r):
        rs = []
        if not r['_Shape']:
            rs.append(REASON_SHAPE_UNMAPPED)
        if pd.isna(r['_Carat']):
            rs.append(REASON_CARAT_BAD)
        elif r['_Shape'] and r['_BucketID'] is None:
            rs.append(REASON_NO_GRID)
        if not r['_cut_pass']:
            rs.append('Cut not Ideal/Excellent')
        if not r['_polish_pass']:
            rs.append('Polish not Ideal/Excellent')
        if not r['_symmetry_pass']:
            rs.append('Symmetry not Ideal/Excellent')
        if not r['_ratio_pass']:
            rs.append(r['_ratio_reason'] or 'Ratio not within standard')
        return rs

    df['_FailReasons'] = df.apply(_fail_reasons, axis=1)
    df['_Eligible'] = df['_FailReasons'].apply(len) == 0
    return df


def allocate(df: pd.DataFrame, req_df: pd.DataFrame, sort_col=None, ascending=True):
    """Greedily allocates eligible stones to requirement buckets, respecting
    'Need Remaining'. Returns (annotated df, updated req_df)."""
    df = df.copy()
    df['_Selected'] = False
    df['_AllocNote'] = ''

    need_remaining = dict(zip(req_df['Bucket ID'], req_df['Need Remaining']))

    eligible = df[df['_Eligible'] & df['_BucketID'].notna()]
    if sort_col and sort_col in df.columns:
        eligible = eligible.sort_values(sort_col, ascending=ascending, na_position='last')

    for idx in eligible.index:
        bid = df.at[idx, '_BucketID']
        if need_remaining.get(bid, 0) > 0:
            df.at[idx, '_Selected'] = True
            need_remaining[bid] -= 1
        else:
            df.at[idx, '_AllocNote'] = REASON_FULFILLED

    req_df = req_df.copy()
    req_df['Need Remaining'] = req_df['Bucket ID'].map(need_remaining)

    def _reason(r):
        if r['_Selected']:
            return ''
        if r['_FailReasons']:
            return '; '.join(r['_FailReasons'])
        return r['_AllocNote'] or REASON_FULFILLED

    df['_ReasonText'] = df.apply(_reason, axis=1)
    df['_StatusText'] = np.where(df['_Selected'], 'SELECTED', 'REJECTED')
    return df, req_df


def split_results(df: pd.DataFrame, original_cols: list, req_df: pd.DataFrame):
    new_names = {'Matched Shape', 'Matched Clarity', 'Matched Color', 'Matched Size Group',
                 'Carat Used', 'Ratio Used', 'Status', 'Reason'}
    df = df.copy()
    safe_original = []
    for c in original_cols:
        if c in new_names:
            new_c = f'{c} (Source)'
            df = df.rename(columns={c: new_c})
            safe_original.append(new_c)
        else:
            safe_original.append(c)

    out_cols = safe_original + [
        '_Shape', '_Clarity', '_Color', '_Carat', '_Ratio', '_StatusText', '_ReasonText',
    ]
    rename = {
        '_Shape': 'Matched Shape', '_Clarity': 'Matched Clarity',
        '_Color': 'Matched Color', '_Carat': 'Carat Used', '_Ratio': 'Ratio Used',
        '_StatusText': 'Status', '_ReasonText': 'Reason',
    }
    clean = df[out_cols].rename(columns=rename)
    bucket_lookup = req_df.set_index('Bucket ID')['Size Group']
    df = df.copy()
    df['Matched Size Group'] = df['_BucketID'].map(bucket_lookup)
    clean.insert(list(clean.columns).index('Matched Color') + 1, 'Matched Size Group',
                 df['Matched Size Group'].values)

    selection = clean[df['_Selected'].values].reset_index(drop=True)
    rejection = clean[~df['_Selected'].values].reset_index(drop=True)
    return selection, rejection
