# f1_data.py (STRICT: skip any race with rain)

import os
import fastf1
import pandas as pd
import numpy as np

from config import SEASONS, RACES, FASTF1_CACHE_DIR
from features import compute_aebi_for_session, build_features_from_laps
from utils import save_df

OUT_PATH = "data/f1_lap_dataset.csv"
PROGRESS_DIR = "data/per_race"


def init_fastf1():
    os.makedirs(FASTF1_CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(FASTF1_CACHE_DIR)


def get_all_race_names_for_season(season: int):
    if RACES is not None:
        return RACES
    try:
        schedule = fastf1.get_event_schedule(season)
        # keep conventional + sprint weekends
        return schedule[schedule['EventFormat'].isin(['conventional', 'sprint'])]['EventName'] \
            .dropna().unique().tolist()
    except Exception as e:
        print(f"Error fetching schedule for {season}: {e}")
        return []


def load_session(season, race_name):
    try:
        session = fastf1.get_session(season, race_name, 'R')
        # weather=True gives you session.weather_data which includes Rainfall (bool) :contentReference[oaicite:3]{index=3}
        session.load(laps=True, telemetry=False, weather=True)
        return session
    except Exception as e:
        print(f"  Could not load session {race_name} {season}: {e}")
        return None


def race_had_rain(session) -> bool:
    """
    Strict rule: if Rainfall was ever True in the race weather feed, skip the entire race.
    FastF1 weather channel includes Rainfall (bool). :contentReference[oaicite:4]{index=4}
    """
    try:
        w = session.weather_data
        if w is None or len(w) == 0:
            return False  # if missing, we cannot prove rain; keep race (or choose to skip if you prefer)
        if 'Rainfall' not in w.columns:
            return False
        return bool(w['Rainfall'].fillna(False).any())
    except Exception:
        return False


def ensure_trackstatus(laps: pd.DataFrame, session) -> pd.DataFrame:
    """
    Ensure TrackStatus exists in laps if FastF1 provided it.
    FastF1 docs: Laps have TrackStatus and you can filter by it. :contentReference[oaicite:5]{index=5}
    """
    if 'TrackStatus' in laps.columns:
        return laps

    # Some versions expose helpers; if not available, we just return laps as-is.
    # We avoid hard failing because data availability varies by season/session.
    try:
        # In FastF1, Laps can be enriched; if this method exists, it will add TrackStatus
        laps = laps.add_track_status()
        return laps
    except Exception:
        return laps


def build_dataset_all_races():
    init_fastf1()
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    all_rows = []

    for season in SEASONS:
        races = get_all_race_names_for_season(season)
        print(f"=== Season {season} ({len(races)} races) ===")

        for race_name in races:
            out_file = os.path.join(PROGRESS_DIR, f"{season}_{race_name.replace(' ', '_')}.csv")

            # Resume logic
            if os.path.exists(out_file):
                print(f"  Skipping (cached): {race_name}")
                try:
                    all_rows.append(pd.read_csv(out_file, low_memory=False))
                except:
                    pass
                continue

            print(f"  Processing: {race_name}")
            session = load_session(season, race_name)
            if session is None:
                continue

            # STRICT: skip any race with any rainfall
            if race_had_rain(session):
                print(f"  ⛈️ Skipping {season} {race_name}: Rainfall detected in weather feed.")
                continue

            laps = session.laps
            if laps is None or laps.empty:
                continue

            laps = ensure_trackstatus(laps, session)

            # AEBI: track-level constant (fine if that is your intent)
            try:
                fastest = laps.pick_fastest()
                tel = fastest.get_telemetry()
                aebi = compute_aebi_for_session(tel)
            except Exception:
                aebi = 0.0

            # Build your features (make sure build_features_from_laps keeps Driver, LapNumber, LapTimeSec, Prev/Next, etc.)
            df_feats = build_features_from_laps(laps, aebi)

            # Make sure TrackStatus makes it into df_feats
            if 'TrackStatus' not in df_feats.columns and 'TrackStatus' in laps.columns:
                join_cols = []
                for c in ['Driver', 'LapNumber']:
                    if c in df_feats.columns and c in laps.columns:
                        join_cols.append(c)

                if len(join_cols) == 2:
                    df_feats = df_feats.merge(
                        laps[['Driver', 'LapNumber', 'TrackStatus']],
                        on=['Driver', 'LapNumber'],
                        how='left'
                    )

            # Metadata
            df_feats['Season'] = season
            df_feats['RaceName'] = race_name
            df_feats['RaceKey'] = df_feats['Season'].astype(str) + "_" + df_feats['RaceName'].astype(str)

            df_feats.to_csv(out_file, index=False)
            all_rows.append(df_feats)

    if not all_rows:
        print("No data found.")
        return

    full_df = pd.concat(all_rows, ignore_index=True)
    save_df(full_df, OUT_PATH)
    print(f"Saved {len(full_df)} laps to {OUT_PATH}")


if __name__ == "__main__":
    build_dataset_all_races()
