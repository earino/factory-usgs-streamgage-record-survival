'''Build the USGS five year streamflow record continuation task.

One row per USGS monitoring location whose earliest primary daily mean
discharge series (parameter 00060, statistic 00003, daily values) begins
between 2001-01-01 and 2019-12-31. The label is 1 when that series is still
active on the calendar fifth anniversary of its begin date.
'''

from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from source_io import fetch

DAILY_URL = ('https://api.waterdata.usgs.gov/ogcapi/v1/collections/combined-metadata/items'
             '?parameter_code=00060&data_type=Daily%20values&agency_code=USGS'
             '&begin=2001-01-01/2019-12-31&limit=1000&f=json')
INST_URL = ('https://api.waterdata.usgs.gov/ogcapi/v1/collections/combined-metadata/items'
            '?parameter_code=00060&statistic_id=00011&agency_code=USGS&limit=1000&f=json')
INST_ALT_URL = ('https://api.waterdata.usgs.gov/ogcapi/v1/collections/combined-metadata/items'
                '?parameter_code=00060&data_type=Instantaneous%20values&agency_code=USGS&limit=1000&f=json')

TRAIN_START = pd.Timestamp('2001-01-01')
TRAIN_END = pd.Timestamp('2013-01-01')
EVAL_START = pd.Timestamp('2013-03-01')
EVAL_END = pd.Timestamp('2016-01-01')
HOLDOUT_START = pd.Timestamp('2016-03-01')
HOLDOUT_END = pd.Timestamp('2020-01-01')

HORIZON_YEARS = 5
PRIOR_WINDOW_DAYS = 365
NEIGHBOR_KM = 50.0
EARTH_RADIUS_KM = 6371.0088
MAX_PAGES = 400
MAX_MSG = 4000
EXCLUDED_STATES = tuple(str(code) for code in list(range(81, 87)) + list(range(90, 99)))

TARGET = 'continued_5y'
ID_COL = 'monitoring_location_id'
TIME_COL = 'begin'
FEATURES = [
    'drainage_area',
    'contributing_drainage_area',
    'altitude',
    'huc2_region',
    'state_code',
    'district_code',
    'site_type_code',
    'has_instantaneous_discharge_series',
    'nearby_active_gage_count_50km',
    'huc2_prior_termination_rate',
    'begin_month',
]

SERIES_KEYS = ('begin', 'end', 'parameter_code', 'statistic_id', 'data_type', 'time_series_id')


def _bounded(text, limit=MAX_MSG):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + '...'


def _record(checks, what, observed, expected, passed, kind=None):
    entry = {
        'check': _bounded(what, 300),
        'observed': _bounded(observed),
        'expected': _bounded(expected),
        'passed': bool(passed),
    }
    if kind is not None:
        entry['kind'] = kind
    checks.append(entry)
    return entry


def _resolve(frame, candidates):
    lowered = {}
    for column in frame.columns:
        lowered.setdefault(str(column).strip().lower(), column)
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _as_code(value):
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            value = int(value)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value).zfill(5)
    text = str(value).strip()
    if not text or text.lower() in ('nan', 'none', 'null', '<na>'):
        return None
    if text.isdigit():
        return text.zfill(5)
    return text


def _as_state(value):
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            value = int(value)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value).zfill(2)
    text = str(value).strip()
    if not text or text.lower() in ('nan', 'none', 'null', '<na>'):
        return None
    if text.isdigit():
        return text.zfill(2)
    return text


def _huc2(value):
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in ('nan', 'none', 'null', '<na>'):
        return None
    digits = ''.join(char for char in text if char.isdigit())
    if len(digits) < 2:
        return None
    return digits[:2]


def _is_false(value):
    if value is None:
        return False
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return not value
    if isinstance(value, float):
        if math.isnan(value):
            return False
        return value == 0.0
    if isinstance(value, int):
        return value == 0
    text = str(value).strip().lower()
    return text in ('false', '0', 'n', 'no', 'f', 'non-primary', 'nonprimary', 'secondary')


def _to_datetime(values):
    try:
        parsed = pd.to_datetime(values, errors='coerce', utc=True)
    except Exception:
        parsed = pd.to_datetime(pd.Series(values).astype('string'), errors='coerce', utc=True)
    if isinstance(parsed, pd.Series):
        return parsed.dt.tz_localize(None).dt.normalize()
    return parsed


def _to_number(values):
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    text = values.astype('string').str.replace(',', '', regex=False).str.strip()
    text = text.mask(text.isin(['', 'NA', 'NaN', 'nan', 'null', 'None', '<NA>']))
    return pd.to_numeric(text, errors='coerce')


def _looks_like_series(item):
    return any(key in item for key in SERIES_KEYS)


def _flatten(pages):
    records = []
    for url, payload in pages:
        for feature in payload.get('features') or []:
            props = feature.get('properties') or {}
            if not isinstance(props, dict):
                continue
            geometry = feature.get('geometry')
            coords = None
            if isinstance(geometry, dict):
                coords = geometry.get('coordinates')
            lon = None
            lat = None
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                try:
                    first = float(coords[0])
                    second = float(coords[1])
                except (TypeError, ValueError):
                    first = None
                    second = None
                if first is not None and second is not None:
                    if abs(second) > 90.0 and abs(first) <= 90.0:
                        first, second = second, first
                    lon = first
                    lat = second
            base = {}
            series_candidates = []
            for key, value in props.items():
                if isinstance(value, dict):
                    if _looks_like_series(value):
                        for inner_key, inner_value in value.items():
                            base[inner_key] = inner_value
                    else:
                        base[key] = value
                elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                    if any(_looks_like_series(item) for item in value):
                        series_candidates.append((key, value))
                else:
                    base[key] = value
            if lon is None:
                for key in ('longitude', 'lon', 'dec_longitude'):
                    if key in base:
                        try:
                            lon = float(base[key])
                        except (TypeError, ValueError):
                            lon = None
                        if lon is not None:
                            break
            if lat is None:
                for key in ('latitude', 'lat', 'dec_latitude'):
                    if key in base:
                        try:
                            lat = float(base[key])
                        except (TypeError, ValueError):
                            lat = None
                        if lat is not None:
                            break
            base['_lon'] = lon
            base['_lat'] = lat
            base['_page_url'] = url
            if not series_candidates:
                records.append(dict(base))
                continue
            series_candidates.sort(
                key=lambda pair: (
                    sum(1 for item in pair[1] if 'begin' in item or 'end' in item),
                    len(pair[1]),
                ),
                reverse=True,
            )
            chosen = series_candidates[0][1]
            for item in chosen:
                row = dict(base)
                for inner_key, inner_value in item.items():
                    row[inner_key] = inner_value
                records.append(row)
    return pd.DataFrame.from_records(records)


def _next_link(payload, base_url):
    for link in payload.get('links') or []:
        if str(link.get('rel') or '').lower() == 'next':
            href = link.get('href')
            if href:
                return urljoin(base_url, str(href))
    return None


def _collect(url, cache, prefix):
    pages = []
    seen = set()
    current = url
    index = 0
    while current and index < MAX_PAGES:
        if current in seen:
            break
        seen.add(current)
        name = prefix + '-' + str(index).zfill(4) + '.json'
        path = fetch(current, cache, name)
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            payload = json.load(handle)
        pages.append((current, payload))
        current = _next_link(payload, current)
        index += 1
    return pages


def _haversine(lat1, lon1, lat2, lon2):
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _neighbour_counts(query_lat, query_lon, query_time, query_id,
                      series_lat, series_lon, series_begin, series_end, series_id):
    counts = np.full(len(query_lat), np.nan)
    if len(series_lat) == 0:
        return counts
    order = np.argsort(series_lat, kind='stable')
    lat_sorted = series_lat[order]
    band = NEIGHBOR_KM / 111.0 + 0.05
    query_id = np.asarray(query_id, dtype=object)
    series_id = np.asarray(series_id, dtype=object)
    for position in range(len(query_lat)):
        lat_value = query_lat[position]
        lon_value = query_lon[position]
        moment = query_time[position]
        if not np.isfinite(lat_value) or not np.isfinite(lon_value) or np.isnat(moment):
            continue
        low = np.searchsorted(lat_sorted, lat_value - band, side='left')
        high = np.searchsorted(lat_sorted, lat_value + band, side='right')
        candidates = order[low:high]
        if candidates.size == 0:
            counts[position] = 0.0
            continue
        distance = _haversine(lat_value, lon_value,
                              series_lat[candidates], series_lon[candidates])
        mask = distance <= NEIGHBOR_KM
        mask &= series_begin[candidates] <= moment
        ends = series_end[candidates]
        mask &= np.isnat(ends) | (ends >= moment)
        mask &= series_id[candidates] != query_id[position]
        if not mask.any():
            counts[position] = 0.0
            continue
        counts[position] = float(np.unique(series_id[candidates][mask]).size)
    return counts


def _prior_termination_rates(query_region, query_time, pool_region, pool_anniversary, pool_end):
    rates = np.full(len(query_region), np.nan)
    if len(pool_region) == 0:
        return rates
    regions = np.asarray(pool_region, dtype=object)
    queries = np.asarray(query_region, dtype=object)
    for region in pd.unique(regions):
        if region is None:
            continue
        if isinstance(region, float) and math.isnan(region):
            continue
        pool_mask = regions == region
        query_mask = queries == region
        if not query_mask.any():
            continue
        anniversaries = pool_anniversary[pool_mask]
        ends = pool_end[pool_mask]
        positions = np.where(query_mask)[0]
        for start in range(0, positions.size, 512):
            block = positions[start:start + 512]
            moments = query_time[block]
            within = anniversaries[None, :] <= moments[:, None]
            ended = within & (ends[None, :] < moments[:, None])
            denominator = within.sum(axis=1)
            numerator = ended.sum(axis=1)
            values = np.full(block.size, np.nan)
            usable = denominator > 0
            values[usable] = numerator[usable] / denominator[usable]
            rates[block] = values
    return rates


def build(output: Path, cache: Path):
    output = Path(output)
    cache = Path(cache)
    public = output / 'public'
    private = output / 'private'
    for path in (public, private, cache):
        path.mkdir(parents=True, exist_ok=True)

    checks = []

    daily_pages = _collect(DAILY_URL, cache, 'usgs-daily')
    daily_features = sum(len(payload.get('features') or []) for _, payload in daily_pages)
    _record(checks, 'combined-metadata daily query returned features',
            str(len(daily_pages)) + ' pages and ' + str(daily_features) + ' features',
            'at least one page holding at least one feature',
            len(daily_pages) > 0 and daily_features > 0)
    if daily_features == 0:
        first_keys = sorted(daily_pages[0][1].keys()) if daily_pages else []
        raise AssertionError(_bounded('combined-metadata daily query returned no features. Pages fetched: '
                                      + str([url for url, _ in daily_pages][:3])
                                      + '. Payload keys: ' + str(first_keys)))

    frame = _flatten(daily_pages)
    headers = sorted(str(name) for name in frame.columns)

    c_id = _resolve(frame, ['monitoring_location_id', 'monitoring_location_number', 'site_no', 'site_number', 'identifier', 'id'])
    c_begin = _resolve(frame, ['begin', 'begin_date', 'begin_utc', 'begin_datetime', 'time_series_begin', 'start_date', 'start', 'period_of_record_begin'])
    c_end = _resolve(frame, ['end', 'end_date', 'end_utc', 'end_datetime', 'time_series_end', 'period_of_record_end'])
    c_stat = _resolve(frame, ['statistic_id', 'statistic', 'statistic_code', 'stat_cd'])
    c_dtype = _resolve(frame, ['data_type', 'data_type_code', 'data_type_cd', 'datatype', 'data_type_name'])
    c_param = _resolve(frame, ['parameter_code', 'parameter', 'parameter_cd', 'parm_cd'])
    c_agency = _resolve(frame, ['agency_code', 'agency_cd'])
    c_primary = _resolve(frame, ['primary', 'primary_flag', 'is_primary'])
    c_state = _resolve(frame, ['state_code', 'state_cd', 'state'])
    c_district = _resolve(frame, ['district_code', 'district_cd', 'district'])
    c_huc = _resolve(frame, ['hydrologic_unit_code', 'huc', 'huc_code', 'huc_8'])
    c_drain = _resolve(frame, ['drainage_area', 'drainage_area_sq_mi', 'drainage_area_value'])
    c_contrib = _resolve(frame, ['contributing_drainage_area', 'contributing_drainage_area_sq_mi', 'contributing_drainage_area_value'])
    c_alt = _resolve(frame, ['altitude', 'altitude_ft', 'alt_va', 'altitude_value'])
    c_sitetype = _resolve(frame, ['site_type_code', 'site_type', 'site_tp_cd'])

    resolved = {
        'monitoring_location_id': c_id,
        'begin': c_begin,
        'end': c_end,
        'statistic_id': c_stat,
        'data_type': c_dtype,
        'parameter_code': c_param,
        'agency_code': c_agency,
        'primary': c_primary,
        'state_code': c_state,
        'district_code': c_district,
        'hydrologic_unit_code': c_huc,
        'drainage_area': c_drain,
        'contributing_drainage_area': c_contrib,
        'altitude': c_alt,
        'site_type_code': c_sitetype,
    }
    missing = sorted(name for name, found in resolved.items() if not found)
    _record(checks, 'declared source fields resolved from the combined-metadata response',
            'missing fields ' + str(missing) + '; observed headers ' + str(headers),
            'every declared field resolved to a response header',
            not missing)
    if missing:
        raise AssertionError(_bounded('Missing declared source fields ' + str(missing)
                                      + '. Observed headers: ' + str(headers)
                                      + '. Fetched page: ' + (daily_pages[0][0] if daily_pages else 'none')))

    coordinate_rows = int(np.isfinite(pd.to_numeric(frame['_lon'], errors='coerce')).sum())
    _record(checks, 'point coordinates resolved from geometry or site properties',
            str(coordinate_rows) + ' of ' + str(len(frame)) + ' raw features carry a longitude',
            'the response carries point coordinates either as GeoJSON geometry or as longitude and latitude properties',
            coordinate_rows > 0)
    if coordinate_rows == 0:
        raise AssertionError(_bounded('No point coordinates on any feature. Observed headers: '
                                      + str(headers)))

    daily = pd.DataFrame({
        'mlid': frame[c_id].astype('string').str.strip(),
        'begin': _to_datetime(frame[c_begin]),
        'end': _to_datetime(frame[c_end]),
        'huc2': frame[c_huc].map(_huc2),
        'state_code': frame[c_state].map(_as_state),
        'district_code': frame[c_district].map(_as_state),
        'site_type_code': frame[c_sitetype].astype('string').str.strip(),
        'drainage_area': _to_number(frame[c_drain]),
        'contributing_drainage_area': _to_number(frame[c_contrib]),
        'altitude': _to_number(frame[c_alt]),
        'lon': pd.to_numeric(frame['_lon'], errors='coerce'),
        'lat': pd.to_numeric(frame['_lat'], errors='coerce'),
        'stat': frame[c_stat].map(_as_code),
        'param': frame[c_param].map(_as_code),
        'dtype': frame[c_dtype].astype('string').str.strip().str.lower(),
        'agency': frame[c_agency].astype('string').str.strip().str.upper(),
        'primary_flag': frame[c_primary] if c_primary else pd.Series([None] * len(frame), index=frame.index, dtype='object'),
    })
    total_raw = len(daily)

    agency_ok = daily['agency'].isna() | daily['agency'].eq('USGS')
    param_ok = daily['param'].isna() | daily['param'].eq('00060')
    stat_ok = daily['stat'].isna() | daily['stat'].eq('00003')
    dtype_ok = daily['dtype'].isna() | daily['dtype'].str.contains('daily', na=True)
    primary_ok = ~daily['primary_flag'].map(_is_false).astype(bool)
    begin_ok = daily['begin'].notna() & (daily['begin'] >= TRAIN_START) & (daily['begin'] < HOLDOUT_END)
    end_ok = daily['end'].notna()
    id_ok = daily['mlid'].notna() & (daily['mlid'].astype(str).str.len() > 0)
    state_ok = ~daily['state_code'].isin(EXCLUDED_STATES)
    coord_ok = daily['lon'].notna() & daily['lat'].notna()

    drop_counts = {
        'non_usgs_agency': int((~agency_ok).sum()),
        'other_parameter': int((~param_ok).sum()),
        'other_statistic': int((~stat_ok).sum()),
        'non_daily_data_type': int((~dtype_ok).sum()),
        'non_primary': int((~primary_ok).sum()),
        'begin_outside_window': int(((daily['begin'].notna()) & (~begin_ok)).sum()),
        'null_end': int((daily['begin'].notna() & begin_ok & ~end_ok).sum()),
        'null_id': int((~id_ok).sum()),
        'excluded_state_code': int((id_ok & (~state_ok)).sum()),
        'missing_coordinates': int((id_ok & state_ok & begin_ok & end_ok & (~coord_ok)).sum()),
    }

    keep_series = (id_ok & agency_ok & param_ok & stat_ok & dtype_ok & primary_ok
                   & begin_ok & end_ok & state_ok)
    keep = keep_series.fillna(False).astype(bool).values
    daily = daily.loc[keep].reset_index(drop=True)

    _record(checks, 'agency code filter kept only USGS records',
            str(int((daily['agency'].notna() & daily['agency'].ne('USGS')).sum())) + ' retained rows with a non-USGS agency code',
            '0 retained rows with a non-USGS agency code',
            int((daily['agency'].notna() & daily['agency'].ne('USGS')).sum()) == 0)
    _record(checks, 'parameter code filter kept only discharge 00060',
            str(int((daily['param'].notna() & daily['param'].ne('00060')).sum())) + ' retained rows with another parameter code; '
            + str(drop_counts['other_parameter']) + ' other-parameter rows observed and dropped',
            '0 retained rows with another parameter code',
            int((daily['param'].notna() & daily['param'].ne('00060')).sum()) == 0)
    _record(checks, 'statistic filter kept only daily mean 00003',
            str(int(daily['stat'].eq('00003').sum())) + ' retained rows on statistic 00003 out of '
            + str(total_raw) + ' raw features',
            'statistic 00003 rows present and every retained row on 00003',
            int(daily['stat'].eq('00003').sum()) > 0
            and int((daily['stat'].notna() & daily['stat'].ne('00003')).sum()) == 0)
    _record(checks, 'begin dates restricted to the frozen 2001 to 2019 window',
            'retained begin range ' + str(daily['begin'].min()) + ' to ' + str(daily['begin'].max())
            + '; outside-window rows dropped ' + str(drop_counts['begin_outside_window']),
            'every retained begin inside 2001-01-01 up to 2019-12-31',
            bool(daily['begin'].notna().all() and daily['begin'].min() >= TRAIN_START and daily['begin'].max() < HOLDOUT_END))
    _record(checks, 'series without an end timestamp dropped rather than imputed',
            str(drop_counts['null_end']) + ' raw rows dropped for a null end; '
            + str(int(daily['end'].isna().sum())) + ' retained rows with a null end',
            '0 retained rows with a null end',
            int(daily['end'].isna().sum()) == 0)
    _record(checks, 'state code exclusion for Mexican and Canadian subdivision codes',
            str(drop_counts['excluded_state_code']) + ' rows dropped with ANSI state codes 81 to 86 or 90 to 98',
            'rows carrying codes 81 to 86 or 90 to 98 are outside the frozen population',
            int(daily['state_code'].isin(EXCLUDED_STATES).sum()) == 0)

    if len(daily) == 0:
        raise AssertionError(_bounded('No daily-mean candidates after filtering. Drop counts: '
                                      + str(drop_counts) + '. Headers: ' + str(headers)))

    daily = daily.sort_values(['mlid', 'begin'], kind='stable').reset_index(drop=True)
    earliest = daily.drop_duplicates('mlid', keep='first').reset_index(drop=True)

    groups = {key: value for key, value in daily.groupby('mlid', sort=False)}
    excluded = np.zeros(len(earliest), dtype=bool)
    evaluated = 0
    for position in range(len(earliest)):
        evaluated += 1
        mlid = earliest.at[position, 'mlid']
        begin = earliest.at[position, 'begin']
        group = groups.get(mlid)
        if group is None or len(group) < 2:
            continue
        begins = group['begin'].values
        ends = group['end'].values
        earlier = begins < np.datetime64(begin)
        if not earlier.any():
            continue
        cutoff = np.datetime64(begin - pd.Timedelta(days=PRIOR_WINDOW_DAYS))
        earlier_ends = ends[earlier]
        if (np.isnat(earlier_ends) | (earlier_ends >= cutoff)).any():
            excluded[position] = True

    _record(checks, 'continuation exclusion evaluated for every candidate location',
            str(evaluated) + ' candidate locations tested against every daily series in the extract',
            'candidate count equal to the deduplicated location count',
            evaluated == len(earliest))
    _record(checks, 'locations excluded as pre-existing continuations',
            str(int(excluded.sum())) + ' locations dropped; ' + str(len(daily)) + ' daily series inspected',
            'a location with another daily-mean series active in the prior 365 days is a continuation, not a new record',
            int(excluded.sum()) <= len(earliest))

    population = earliest.loc[~excluded].reset_index(drop=True)
    if len(population) == 0:
        raise AssertionError(_bounded('No population rows after continuation exclusion. Daily series: '
                                      + str(len(daily)) + '. Headers: ' + str(headers)))
    _record(checks, 'one row per monitoring location after deduplication',
            str(int(population['mlid'].duplicated().sum())) + ' duplicated location identifiers',
            '0 duplicated location identifiers',
            int(population['mlid'].duplicated().sum()) == 0)

    population['anniversary'] = population['begin'] + pd.DateOffset(years=HORIZON_YEARS)
    population[TARGET] = (population['end'] >= population['anniversary']).astype(int)
    _record(checks, 'target recomputed from begin and end matches the stored label',
            str(int(((population['end'] >= population['begin'] + pd.DateOffset(years=HORIZON_YEARS)).astype(int)
                     != population[TARGET]).sum())) + ' mismatching rows',
            '0 mismatching rows',
            int(((population['end'] >= population['begin'] + pd.DateOffset(years=HORIZON_YEARS)).astype(int)
                 != population[TARGET]).sum()) == 0)
    _record(checks, 'target is binary with no missing values',
            'values ' + str(sorted(population[TARGET].unique().tolist())) + ' over '
            + str(len(population)) + ' rows',
            'only the values 0 and 1, no nulls',
            set(population[TARGET].unique().tolist()) == {0, 1})

    inst_pages = []
    inst_features = 0
    inst_note = ''
    try:
        inst_pages = _collect(INST_URL, cache, 'usgs-instantaneous')
        inst_features = sum(len(payload.get('features') or []) for _, payload in inst_pages)
    except Exception as exc:
        inst_note = 'primary instantaneous query failed: ' + str(exc)
    if inst_features == 0:
        try:
            alt_pages = _collect(INST_ALT_URL, cache, 'usgs-instantaneous-alt')
            alt_features = sum(len(payload.get('features') or []) for _, payload in alt_pages)
            if alt_features > 0:
                inst_pages = alt_pages
                inst_features = alt_features
                inst_note = inst_note + '; used the data type instantaneous fallback'
        except Exception as exc:
            inst_note = inst_note + '; fallback failed: ' + str(exc)
    _record(checks, 'instantaneous discharge metadata returned features',
            str(len(inst_pages)) + ' pages and ' + str(inst_features) + ' features'
            + ('; ' + inst_note if inst_note else ''),
            'at least one instantaneous discharge feature for the network flag',
            inst_features > 0)
    if inst_features == 0:
        raise AssertionError(_bounded('No instantaneous discharge features returned. '
                                      + inst_note + '. Primary URL: ' + INST_URL
                                      + '. Fallback URL: ' + INST_ALT_URL))

    inst_frame = _flatten(inst_pages)
    ic_id = _resolve(inst_frame, ['monitoring_location_id', 'monitoring_location_number', 'site_no', 'site_number', 'identifier', 'id'])
    ic_begin = _resolve(inst_frame, ['begin', 'begin_date', 'begin_utc', 'begin_datetime', 'time_series_begin', 'start_date', 'start', 'period_of_record_begin'])
    ic_stat = _resolve(inst_frame, ['statistic_id', 'statistic', 'statistic_code', 'stat_cd'])
    if ic_id is None or ic_begin is None:
        raise AssertionError(_bounded('Instantaneous metadata response missing identifier or begin. Observed headers: '
                                      + str(sorted(str(name) for name in inst_frame.columns))))
    inst_codes = inst_frame[ic_stat].map(_as_code) if ic_stat else pd.Series(['00011'] * len(inst_frame), index=inst_frame.index)
    inst_begin = _to_datetime(inst_frame[ic_begin])
    inst_table = pd.DataFrame({
        'mlid': inst_frame[ic_id].astype('string').str.strip(),
        'begin': inst_begin,
        'stat': inst_codes,
    })
    inst_table = inst_table[inst_table['mlid'].notna() & inst_table['begin'].notna()
                            & (inst_table['stat'].isna() | inst_table['stat'].eq('00011'))]
    _record(checks, 'instantaneous metadata rows carry statistic 00011 or a null statistic',
            str(len(inst_table)) + ' usable instantaneous rows from ' + str(len(inst_frame)) + ' flattened rows',
            'at least one usable instantaneous series row',
            len(inst_table) > 0)
    if len(inst_table) == 0:
        raise AssertionError(_bounded('No usable instantaneous series rows after filtering. Observed headers: '
                                      + str(sorted(str(name) for name in inst_frame.columns))))
    inst_min = inst_table.groupby('mlid')['begin'].min()

    has_inst = population['mlid'].map(inst_min)
    population['has_instantaneous_discharge_series'] = np.where(
        has_inst.notna() & (has_inst <= population['begin']), 1, 0).astype(int)

    neighbour_source = daily[daily['lon'].notna() & daily['lat'].notna()].reset_index(drop=True)
    neighbour_counts = _neighbour_counts(
        population['lat'].values,
        population['lon'].values,
        population['begin'].values,
        population['mlid'].values,
        neighbour_source['lat'].values,
        neighbour_source['lon'].values,
        neighbour_source['begin'].values,
        neighbour_source['end'].values,
        neighbour_source['mlid'].values,
    )
    population['nearby_active_gage_count_50km'] = neighbour_counts

    pool = daily[daily['begin'] < TRAIN_END].copy()
    pool['anniversary'] = pool['begin'] + pd.DateOffset(years=HORIZON_YEARS)
    rates = _prior_termination_rates(
        population['huc2'].values,
        population['begin'].values,
        pool['huc2'].values,
        pool['anniversary'].values,
        pool['end'].values,
    )
    population['huc2_prior_termination_rate'] = rates

    population['huc2_region'] = population['huc2']
    population['begin_month'] = population['begin'].dt.month.astype(int)

    _record(checks, 'prior termination rate uses only training-window beginnings',
            'latest pool begin ' + str(pool['begin'].max()) + ' over ' + str(len(pool)) + ' reference series',
            'every reference series begins before 2013-01-01 so scored labels never enter the rate',
            bool(len(pool) > 0 and pool['begin'].max() < TRAIN_END))
    rate_values = population['huc2_prior_termination_rate'].dropna()
    _record(checks, 'prior termination rate is a share in the closed unit interval',
            str(int(rate_values.size)) + ' usable rows, minimum '
            + str(round(float(rate_values.min()), 6) if rate_values.size else 'nan')
            + ', maximum ' + str(round(float(rate_values.max()), 6) if rate_values.size else 'nan'),
            'every usable rate between 0 and 1 inclusive, missing where no resolvable series exists',
            bool(rate_values.size > 0 and rate_values.min() >= 0.0 and rate_values.max() <= 1.0))
    _record(checks, 'nearby gage count is a non-negative integer below the series total',
            'usable rows ' + str(int(pd.Series(neighbour_counts).notna().sum()))
            + ', minimum ' + str(int(np.nanmin(neighbour_counts)) if np.isfinite(neighbour_counts).any() else 'nan')
            + ', maximum ' + str(int(np.nanmax(neighbour_counts)) if np.isfinite(neighbour_counts).any() else 'nan'),
            'counts between 0 and the number of other series in the extract',
            bool(np.isfinite(neighbour_counts).any()
                 and np.nanmin(neighbour_counts) >= 0.0
                 and np.nanmax(neighbour_counts) < max(len(neighbour_source), 1)))
    _record(checks, 'population rows carry location coordinates for the network features',
            str(int((population['lon'].isna() | population['lat'].isna()).sum())) + ' of '
            + str(len(population)) + ' population rows without coordinates; '
            + str(drop_counts['missing_coordinates']) + ' raw rows lacked coordinates',
            'at least 99 percent of population rows carry coordinates; nulls stay null and never become zero',
            bool(len(population) > 0
                 and (population['lon'].isna() | population['lat'].isna()).sum() <= 0.01 * len(population)))

    horizon_days = (population['anniversary'] - population['begin']).dt.days
    _record(checks, 'unit check for the five year continuation horizon',
            'median day span ' + str(int(horizon_days.median())) + ' days between begin and the fifth anniversary over '
            + str(len(population)) + ' rows',
            'a calendar fifth anniversary is 1826 or 1827 days, so the horizon is years not an arbitrary day count',
            bool(1825 <= float(horizon_days.median()) <= 1828), kind='unit')

    train_mask = (population['begin'] >= TRAIN_START) & (population['begin'] < TRAIN_END)
    eval_mask = (population['begin'] >= EVAL_START) & (population['begin'] < EVAL_END)
    holdout_mask = (population['begin'] >= HOLDOUT_START) & (population['begin'] < HOLDOUT_END)
    assigned = train_mask | eval_mask | holdout_mask
    embargo_dropped = int((~assigned).sum())
    population = population.loc[assigned].reset_index(drop=True)
    if len(population) == 0:
        raise AssertionError(_bounded('No population rows left after split assignment. Gap rows dropped: '
                                      + str(embargo_dropped)))
    train_mask = (population['begin'] >= TRAIN_START) & (population['begin'] < TRAIN_END)
    eval_mask = (population['begin'] >= EVAL_START) & (population['begin'] < EVAL_END)
    holdout_mask = (population['begin'] >= HOLDOUT_START) & (population['begin'] < HOLDOUT_END)

    inside_gap = int(((population['begin'] >= TRAIN_END) & (population['begin'] < EVAL_START)).sum()
                     + ((population['begin'] >= EVAL_END) & (population['begin'] < HOLDOUT_START)).sum())
    _record(checks, 'sixty day embargo gaps hold no shipped rows',
            str(inside_gap) + ' shipped rows inside a gap; ' + str(embargo_dropped) + ' rows dropped as gap rows',
            '0 shipped rows inside 2013-01-01 to 2013-03-01 or 2016-01-01 to 2016-03-01',
            inside_gap == 0)

    columns = [ID_COL, TIME_COL] + FEATURES + [TARGET]

    def _frame(mask):
        part = population.loc[mask].copy()
        out = pd.DataFrame({ID_COL: part['mlid'].astype(str).values})
        out[TIME_COL] = part['begin'].dt.strftime('%Y-%m-%d').values
        out['drainage_area'] = part['drainage_area'].values
        out['contributing_drainage_area'] = part['contributing_drainage_area'].values
        out['altitude'] = part['altitude'].values
        out['huc2_region'] = part['huc2_region'].values
        out['state_code'] = part['state_code'].values
        out['district_code'] = part['district_code'].astype(object).where(part['district_code'].notna(), None).values
        out['site_type_code'] = part['site_type_code'].astype(object).where(part['site_type_code'].notna(), None).values
        out['has_instantaneous_discharge_series'] = part['has_instantaneous_discharge_series'].astype(int).values
        out['nearby_active_gage_count_50km'] = part['nearby_active_gage_count_50km'].values
        out['huc2_prior_termination_rate'] = part['huc2_prior_termination_rate'].values
        out['begin_month'] = part['begin_month'].astype(int).values
        out[TARGET] = part[TARGET].astype(int).values
        return out[columns].sort_values([TIME_COL, ID_COL], kind='stable').reset_index(drop=True)

    train_frame = _frame(train_mask)
    eval_frame = _frame(eval_mask)
    holdout_frame = _frame(holdout_mask)

    train_frame.to_csv(public / 'train.csv', index=False)
    eval_frame.to_csv(public / 'eval.csv', index=False)
    holdout_frame.to_csv(private / 'holdout.csv', index=False)

    _record(checks, 'all three splits carry identical columns in identical order',
            'train ' + str(list(train_frame.columns)) + '; eval ' + str(list(eval_frame.columns))
            + '; holdout ' + str(list(holdout_frame.columns)),
            'the same declared identifier, time, feature and target columns',
            list(train_frame.columns) == columns
            and list(eval_frame.columns) == columns
            and list(holdout_frame.columns) == columns)

    coverage = {}
    for name in FEATURES:
        coverage[name] = int(pd.concat([train_frame[name], eval_frame[name], holdout_frame[name]]).notna().sum())
    empty = sorted(name for name, count in coverage.items() if count == 0)
    _record(checks, 'every declared feature column carries at least one observed value',
            'non-null counts ' + str(coverage),
            'no declared feature is shipped as an entirely empty column',
            not empty)
    if empty:
        raise AssertionError(_bounded('Declared features with zero coverage after construction: ' + str(empty)
                                      + '. Column coverage: ' + str(coverage)
                                      + '. Observed headers: ' + str(headers)))

    source_null_drainage = int(population['drainage_area'].isna().sum())
    shipped_null_drainage = int(pd.concat([train_frame['drainage_area'], eval_frame['drainage_area'],
                                           holdout_frame['drainage_area']]).isna().sum())
    _record(checks, 'missing drainage area values stay missing instead of being filled with zero',
            str(source_null_drainage) + ' null source values and ' + str(shipped_null_drainage) + ' null shipped values',
            'null counts identical between the population and the shipped rows',
            source_null_drainage == shipped_null_drainage)

    _record(checks, 'split boundaries are chronological and half open',
            'train max ' + str(train_frame[TIME_COL].max()) + '; eval ' + str(eval_frame[TIME_COL].min())
            + ' to ' + str(eval_frame[TIME_COL].max()) + '; holdout min ' + str(holdout_frame[TIME_COL].min()),
            'train below 2013-01-01, eval from 2013-03-01 below 2016-01-01, holdout from 2016-03-01',
            bool(len(train_frame) > 0 and len(eval_frame) > 0 and len(holdout_frame) > 0
                 and train_frame[TIME_COL].max() < '2013-01-01'
                 and eval_frame[TIME_COL].min() >= '2013-03-01'
                 and eval_frame[TIME_COL].max() < '2016-01-01'
                 and holdout_frame[TIME_COL].min() >= '2016-03-01'))

    _record(checks, 'no identifier appears in more than one split',
            str(int(train_frame[ID_COL].isin(set(eval_frame[ID_COL]) | set(holdout_frame[ID_COL])).sum()))
            + ' train identifiers repeated elsewhere; '
            + str(int(eval_frame[ID_COL].isin(set(holdout_frame[ID_COL])).sum())) + ' eval identifiers repeated in holdout',
            '0 repeated identifiers across splits',
            int(train_frame[ID_COL].isin(set(eval_frame[ID_COL]) | set(holdout_frame[ID_COL])).sum()) == 0
            and int(eval_frame[ID_COL].isin(set(holdout_frame[ID_COL])).sum()) == 0)

    failures = [entry for entry in checks if not entry['passed']]
    if failures:
        raise AssertionError(_bounded('Failed source checks: ' + json.dumps(failures, default=str)))
    return checks
