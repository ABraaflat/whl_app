"""
WHL Simulator, Pick'em & Betting Tracker  (whl_app.py)  ·  version 6
=====================================================================
Simulates any Western Hockey League game thousands of times from how both
teams have really played lately (the WHL's own game sheets), then turns the
simulations into:

  * win / score / overtime chances and a watchable animated game
    (with the home team's logo at center ice)
  * projections and prop probabilities for every player (shots, points,
    goals, assists, power-play points, goalie saves)
  * fair odds and a log of your own bets that grades itself after the games
  * a tracker that checks the model's game AND player predictions
  * live win probability, the rest of the season and the WHL playoffs
  * picks of the day, a what-if lab, travel & fatigue, scoring and title races,
    rivalries, and Beat the Bot (a pick'em game against the model, friends welcome)

Run it (in PowerShell, NOT the Python >>> prompt):
    python -m pip install -U streamlit plotly pandas numpy requests
    python -m streamlit run whl_app.py

Data comes from the WHL's public stats feed (HockeyTech, the system behind whl.ca).
"""

import base64
import html as html_lib
import datetime as dt
import io
import json
import math
import os
import re
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

HT_URL = "https://lscluster.hockeytech.com/feed/index.php"
HT_KEY = "f1aa699db3d81487"          # public key used by the WHL's own website
HT_CLIENT = "whl"
HT_LEAGUE_ID = 26                     # Western Hockey League (27 = CHL, 29 = WHL Cup)
HT_SITE_ID = 0

st.set_page_config(page_title="WHL Sim & Bet Tracker", page_icon="🏒", layout="wide",
                   initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------
# Model constants (league-average priors; real team data pulls away from these)
# --------------------------------------------------------------------------
STATES = ("lead", "tied", "trail", "pp", "sh")
MIRROR = {"lead": "trail", "tied": "tied", "trail": "lead", "pp": "sh", "sh": "pp"}
LG_EV = 29.5                      # even-strength shots on goal per 60 (one team)
SCORE_MULT = {"lead": 0.90, "tied": 1.00, "trail": 1.12}   # score effects
LG_RATE = {"lead": LG_EV * 0.90, "tied": LG_EV, "trail": LG_EV * 1.12,
           "pp": 54.0, "sh": 8.0}
SHRINK_MIN = {"lead": 80, "tied": 80, "trail": 80, "pp": 40, "sh": 60}
LG_PEN60 = 3.0                    # minor penalties per team per 60
LG_SH_PCT = 0.100                 # league shooting % on shots on goal
LG_SV = 0.895                     # league save %
PP_GOAL_MULT = 1.30               # power-play shots are more dangerous
HOME_EDGE = 0.040                 # home-ice edge
OT3_SHOT, OT3_GOAL = 1.6, 1.35    # 3-on-3 overtime is wide open
PULL_DOWN1, PULL_DOWN2 = 150, 210 # seconds left when the goalie gets pulled
PULL_SHOT, PULL_GOAL = 1.7, 1.2   # 6-on-5 attack boost
EN_RATE = 0.0045                  # empty-net goals per second against a pulled goalie
DT = 5                            # simulation time step (seconds)
HALF_LIFE = 10                    # recent form: a game 10 games ago counts half as much as the last one
B2B_FOR, B2B_AG = 0.97, 1.03      # second night of a back-to-back: fewer shots for, more against

# Shot-on-goal scoring rate by distance from the net (prior; refit from real shots)
DIST_EDGES = np.array([0, 10, 20, 30, 40, 60, 999.0])
DIST_PRIOR = np.array([0.21, 0.15, 0.095, 0.065, 0.04, 0.018])

# Filled in from the live standings (WHL: B.C. & U.S. = Western, Central & East = Eastern)
CONF_DIVS = {"Western": ("B.C.", "U.S."), "Eastern": ("Central", "East")}
CONF_OF = {d: c for c, ds in CONF_DIVS.items() for d in ds}

PALETTE = ["#1f4e9c", "#c8102e", "#006847", "#f47a38", "#6f263d", "#00843d", "#b4975a", "#002f87",
           "#cf0a2c", "#154734", "#7a3e9d", "#0b7285", "#e07a00", "#333333", "#8a1538", "#2e86de",
           "#556b2f", "#a0522d", "#4b0082", "#008080", "#b22222", "#2f4f4f", "#d4a017", "#5c4033"]
TEAM_COLORS = {}


def team_colors(home, away):
    """Home/away colours, swapping the away colour if the two are too close."""
    def rgb(h):
        h = h.lstrip("#")
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)])
    ch, ca = TEAM_COLORS.get(home, "#1f77b4"), TEAM_COLORS.get(away, "#d62728")
    if np.abs(rgb(ch) - rgb(ca)).sum() < 100:
        ca = "#8A8D8F" if np.abs(rgb(ch) - rgb("#8A8D8F")).sum() > 150 else "#E07A00"
    return ch, ca


# --------------------------------------------------------------------------
# Data fetching
# --------------------------------------------------------------------------
@st.cache_resource
def _session():
    s = requests.Session()
    s.headers["User-Agent"] = "whl-sim-app/1.0"
    return s


@st.cache_resource
def _pbp_store():
    """Finished games never change, so keep their play-by-play for the whole session."""
    return {}


def _clean_jsonp(text):
    """HockeyTech wraps some answers in a JSONP callback; strip it."""
    text = text.strip()
    text = re.sub(r"^angular\.callbacks\._\d+\(", "", text)
    text = re.sub(r"^[a-zA-Z_][a-zA-Z0-9_]*\(", "", text)
    if text.startswith("("):
        text = text[1:]
    text = re.sub(r"\);?\s*$", "", text)
    return text.strip()


def ht_get(params, session=None):
    """One request to the WHL's HockeyTech feed."""
    s = session or _session()
    r = s.get(HT_URL, params={"key": HT_KEY, "client_code": HT_CLIENT, **params}, timeout=25)
    r.raise_for_status()
    text = _clean_jsonp(r.text)
    return json.loads(text) if text else {}


def _sections(data):
    """HockeyTech 'statviewfeed' tables: [{sections: [{title, data: [{row, prop}]}]}]."""
    if isinstance(data, list) and data and isinstance(data[0], dict) and "sections" in data[0]:
        return data[0].get("sections", [])
    if isinstance(data, dict) and "sections" in data:
        return data.get("sections", [])
    return []


def _rows(data):
    out = []
    for sec in _sections(data):
        for item in sec.get("data", []):
            out.append(item.get("row", item) if isinstance(item, dict) else item)
    if not out and isinstance(data, dict):
        sk = data.get("SiteKit") or {}
        for v in sk.values():
            if isinstance(v, list):
                out.extend(x for x in v if isinstance(x, dict))
    return out


def _num(v, default=0.0):
    try:
        if v in (None, "", "-"):
            return default
        return float(str(v).replace("%", "").strip())
    except ValueError:
        return default


def _fix_url(u):
    u = str(u or "").strip()
    if u.startswith("//"):
        u = "https:" + u
    return u.replace("http://", "https://", 1)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def bootstrap():
    d = ht_get(dict(feed="statviewfeed", view="bootstrap", season="latest", pageName="scorebar",
                    league_id=HT_LEAGUE_ID, site_id=HT_SITE_ID))
    if isinstance(d, dict) and isinstance(d.get(HT_CLIENT), dict):
        d = d[HT_CLIENT]
    return d if isinstance(d, dict) else {}


def season_id():
    b = bootstrap()
    sid = b.get("current_season_id")
    regs = [s for s in b.get("regularSeasons", []) if str(s.get("id", "")).isdigit()]
    if regs:
        reg_ids = sorted(int(s["id"]) for s in regs)
        # outside the regular season the "current" id can be a playoff/preseason one
        if not sid or int(sid) not in reg_ids:
            sid = reg_ids[-1] if not sid else max([i for i in reg_ids if i <= int(sid)] or reg_ids)
    return str(sid or "289")


def prev_season_id():
    regs = sorted(int(s["id"]) for s in bootstrap().get("regularSeasons", []) if str(s.get("id", "")).isdigit())
    cur = int(season_id())
    older = [i for i in regs if i < cur]
    return str(older[-1]) if older else None


def season_start_year(sid):
    for s in bootstrap().get("regularSeasons", []) + bootstrap().get("seasons", []):
        if str(s.get("id")) == str(sid):
            for key in ("start_date", "startDate"):
                if str(s.get(key, ""))[:4].isdigit():
                    return int(str(s[key])[:4])
            name = str(s.get("name", ""))
            m = re.search(r"(20\d\d)", name)
            if m:
                return int(m.group(1))
    today = dt.date.today()
    return today.year if today.month >= 8 else today.year - 1


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def teams_info():
    """{team code: {id, name, logo}} for every WHL team."""
    b = bootstrap()
    teams = b.get("teamsNoAll") or [t for t in b.get("teams", []) if str(t.get("id", "")) not in ("", "-1", "0")]
    out = {}
    for t in teams:
        code = str(t.get("team_code") or t.get("code") or "").upper().strip()
        if not code:
            continue
        name = t.get("name") or " ".join(x for x in (t.get("city"), t.get("nickname")) if x) or code
        divs = {str(d.get("id")): re.sub(r"\s*Division$", "", str(d.get("name", ""))) for d in b.get("divisions", [])}
        out[code] = dict(id=str(t.get("id")), name=name, division=divs.get(str(t.get("division_id")), ""),
                         logo=_fix_url(t.get("logo") or t.get("team_logo_url") or t.get("logo_url")))
    return out


def team_logo(code):
    return teams_info().get(code, {}).get("logo", "")


def _id_to_code():
    return {v["id"]: k for k, v in teams_info().items()}


TZ_OFFSET = {"PST": -8, "PDT": -7, "PCT": -8, "MST": -7, "MDT": -6, "MCT": -7, "CST": -6, "CDT": -5, "CCT": -6,
             "EST": -5, "EDT": -4, "ECT": -5}
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                                      "Sep", "Oct", "Nov", "Dec"], 1)}


def _parse_game_date(row, start_year):
    for key in ("date_played", "GameDateISO8601", "game_date", "date"):
        v = str(row.get(key) or "")
        if re.match(r"\d{4}-\d\d-\d\d", v):
            return v[:10]
    v = str(row.get("date_with_day") or row.get("date_time") or "")
    m = re.search(r"([A-Z][a-z]{2})\w*\.?\s+(\d{1,2})", v)
    if not m or m.group(1) not in MONTHS:
        return ""
    mon, day = MONTHS[m.group(1)], int(m.group(2))
    year = start_year if mon >= 8 else start_year + 1
    return f"{year}-{mon:02d}-{day:02d}"


def _start_utc(date_str, status):
    """Start time in UTC from a status like '7:05 pm PST' (7 pm Pacific if unknown)."""
    if not date_str:
        return ""
    m = re.search(r"(\d{1,2}):(\d\d)\s*([ap]m)\s*([A-Z]{3})?", str(status), re.I)
    hh, mm, off = 19, 0, -8
    if m:
        hh = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
        mm = int(m.group(2))
        off = TZ_OFFSET.get((m.group(4) or "PST").upper(), -8)
    local = dt.datetime.fromisoformat(date_str).replace(hour=hh, minute=mm)
    return (local - dt.timedelta(hours=off)).strftime("%Y-%m-%dT%H:%M:%SZ")


@st.cache_data(ttl=900, show_spinner=False)
def season_schedule(sid):
    """Every game of one WHL season as simple dicts."""
    data = ht_get(dict(feed="statviewfeed", view="schedule", season=sid, team="all", month=-1,
                       league_id=HT_LEAGUE_ID, site_id=HT_SITE_ID))
    id2code = _id_to_code()
    start_year = season_start_year(sid)
    games = []
    for sec in _sections(data):
        for item in sec.get("data", []):
            row, prop = item.get("row", {}), item.get("prop", {})
            gid = row.get("game_id") or (prop.get("game_summary") or {}).get("gameLink")
            hid = str((prop.get("home_team_city") or {}).get("teamLink") or row.get("home_team") or "")
            aid = str((prop.get("visiting_team_city") or {}).get("teamLink") or row.get("visiting_team") or "")
            home = id2code.get(hid) or str(row.get("home_team_code") or "").upper()
            away = id2code.get(aid) or str(row.get("visiting_team_code") or "").upper()
            if not gid or not home or not away:
                continue
            status = str(row.get("game_status") or "")
            date = _parse_game_date(row, start_year)
            low = status.lower()
            if low.startswith("final"):
                state = "FINAL"
            elif "postpon" in low or "cancel" in low:
                continue
            elif re.search(r"\d{1,2}:\d\d\s*[ap]m", low) or not low.strip():
                state = "FUT"
            else:
                state = "LIVE"
            end = "SO" if "so" in low.split() else ("OT" if "ot" in low else "REG")
            games.append(dict(id=int(gid), date=date, start=_start_utc(date, status), state=state,
                              home=home, away=away, home_id=hid or teams_info().get(home, {}).get("id"),
                              away_id=aid or teams_info().get(away, {}).get("id"),
                              hs=int(_num(row.get("home_goal_count"), 0)), as_=int(_num(row.get("visiting_goal_count"), 0)),
                              end=end))
    return sorted(games, key=lambda g: (g["date"], g["start"], g["id"]))


def all_games():
    games = list(season_schedule(season_id()))
    prev = prev_season_id()
    if prev:
        try:
            games = season_schedule(prev) + games
        except Exception:
            pass
    return games


def game_index():
    return {g["id"]: g for g in all_games()}


@st.cache_data(ttl=1800, show_spinner=False)
def load_standings():
    info = teams_info()
    rows = []
    try:
        data = ht_get(dict(feed="statviewfeed", view="teams", season=season_id(), league_id=HT_LEAGUE_ID,
                           site_id=HT_SITE_ID, groupTeamsBy="division", context="overall", special="false"))
        for sec in _sections(data):
            hdr = ((sec.get("headers") or {}).get("name") or {}).get("properties") or {}
            div = re.sub(r"\s*Division$", "", str(sec.get("title") or hdr.get("title") or hdr.get("label") or ""))
            for item in sec.get("data", []):
                r = item.get("row", item)
                code = str(r.get("team_code") or "").upper()
                if not code:
                    continue
                div = div or info.get(code, {}).get("division", "")
                otl = int(_num(r.get("ot_losses"))) + int(_num(r.get("shootout_losses")))
                rows.append(dict(
                    team=code, name=info.get(code, {}).get("name") or re.sub(r"^[a-z\*]+\s*-\s*", "", str(r.get("name", code))),
                    gp=int(_num(r.get("games_played"))), w=int(_num(r.get("wins"))), l=int(_num(r.get("losses"))),
                    otl=otl, pts=int(_num(r.get("points"))), gf=int(_num(r.get("goals_for"))),
                    ga=int(_num(r.get("goals_against"))), l10=str(r.get("past_10") or ""), streak=str(r.get("streak") or ""),
                    rw=int(_num(r.get("regulation_wins"), _num(r.get("wins")))), div=div))
    except Exception:
        rows = []
    if not rows:     # standings unavailable: still list every team so the simulator works
        rows = [dict(team=c, name=v["name"], gp=0, w=0, l=0, otl=0, pts=0, gf=0, ga=0, l10="", streak="", rw=0,
                     div="") for c, v in info.items()]
    df = pd.DataFrame(rows).drop_duplicates("team").sort_values("team").reset_index(drop=True)
    df["conf"] = df["div"].map(conference_of)
    return df


def conference_of(div):
    if not div:
        return ""
    try:
        b = bootstrap()
        confs = {str(c.get("conference_id")): re.sub(r"\s*Conference$", "", str(c.get("conference_name", "")))
                 for c in b.get("conferences", [])}
        for d in b.get("divisions", []):
            if re.sub(r"\s*Division$", "", str(d.get("name", ""))) == div and confs.get(str(d.get("conference_id"))):
                return confs[str(d.get("conference_id"))]
    except Exception:
        pass
    d = str(div).lower()
    return "Eastern" if ("east" in d or "central" in d) else "Western"


@st.cache_data(ttl=3600, show_spinner=False)
def league_players():
    """All WHL skaters and goalies this season, keyed by team code."""
    sid = season_id()
    base = dict(feed="statviewfeed", view="players", season=sid, team="all", league_id=HT_LEAGUE_ID,
                site_id=HT_SITE_ID, start=0, limit=2000)
    def rows_with_names(data):
        out = []
        for sec in _sections(data):
            for item in sec.get("data", []):
                r = dict(item.get("row", {}))
                for v in (item.get("prop") or {}).values():
                    if isinstance(v, dict) and v.get("seoName"):
                        r["_full_name"] = v["seoName"]
                out.append(r)
        return out or _rows(data)

    sk_rows = rows_with_names(ht_get(base))
    try:
        gk_rows = rows_with_names(ht_get({**base, "position": "goalies", "qualified": "all"}))
    except Exception:
        gk_rows = []

    def nice(name):
        name = str(name or "").strip()
        if ", " in name:
            last, first = name.split(", ", 1)
            name = f"{first} {last}"
        return name

    skaters, goalies = {}, {}
    for r in sk_rows:
        pos = str(r.get("position") or "F").upper()
        if pos in ("G", "GOALIE"):
            continue
        gp = int(_num(r.get("games_played")))
        if gp <= 0:
            continue
        g, a = _num(r.get("goals")), _num(r.get("assists"))
        shp = _num(r.get("shooting_percentage"))
        shots = _num(r.get("shots"), round(g * 100 / shp) if shp > 0 else 0)   # WHL lists SH%, not shots
        pts_pg, s_pg = (g + a) / gp, shots / gp
        skaters.setdefault(str(r.get("team_code") or "").upper(), []).append(dict(
            id=str(r.get("player_id") or r.get("id")), name=r.get("_full_name") or nice(r.get("name")),
            pos="D" if pos.startswith("D") else ("C" if pos.startswith("C") else pos[:1] or "F"),
            gp=gp, g=int(g), a=int(a), shots=int(shots), ppg=int(_num(r.get("power_play_goals"))),
            # WHL stats have no ice time, so estimate it from production (used only to rank lines)
            toi=float(700 + 260 * pts_pg + 110 * s_pg + (220 if pos.startswith("D") else 0)),
            pim=int(_num(r.get("penalty_minutes"))), pts=int(g + a), num=str(r.get("jersey_number") or ""),
            headshot=f"https://assets.leaguestat.com/whl/240x240/{r.get('player_id') or r.get('id')}.jpg"))
    for r in gk_rows:
        gp = int(_num(r.get("games_played")))
        if gp <= 0:
            continue
        svp = _num(r.get("save_percentage") or r.get("savepct") or r.get("save_pct"))
        if svp > 1:
            svp /= 100 if svp > 10 else 1
        sa = _num(r.get("shots_against") or r.get("shots"))
        ga = _num(r.get("goals_against"))
        saves = _num(r.get("saves"), sa - ga if sa else 0)
        if not sa and ga and svp:
            sa = ga / max(1 - svp, 1e-3)
            saves = sa - ga
        if not svp and sa:
            svp = saves / sa
        mins = _num(r.get("minutes_played") or r.get("minutes"))
        gs = int(_num(r.get("games_started") or r.get("starts") or r.get("gs"), round(mins / 60) if mins else gp))
        goalies.setdefault(str(r.get("team_code") or "").upper(), []).append(dict(
            id=str(r.get("player_id") or r.get("id")), name=r.get("_full_name") or nice(r.get("name") or r.get("shortname")),
            gp=gp, gs=gs,
            sa=int(sa), saves=int(saves), svp=float(svp), gaa=_num(r.get("goals_against_average") or r.get("gaa")),
            w=int(_num(r.get("wins"))), num=str(r.get("jersey_number") or ""), headshot=f"https://assets.leaguestat.com/whl/240x240/{r.get('player_id') or r.get('id')}.jpg"))
    return skaters, goalies


def team_players(team):
    try:
        sk, gk = league_players()
    except Exception:
        return [], []
    return sk.get(team, []), gk.get(team, [])


def _pid(e, *keys):
    """First player id found under any of these keys (plain or nested {player_id})."""
    for k in keys:
        v = e.get(k)
        if isinstance(v, dict):
            v = v.get("player_id") or v.get("id")
        if v not in (None, "", "0", 0):
            return str(v)
    return None


def slim_whl_summary(gs, g):
    """Turn a WHL game summary into the compact play list the simulator uses.

    The WHL publishes goals, penalties, lineups and each goalie's shots against,
    but not the time of each shot. So the real goal/penalty timeline gives the
    score and strength at every moment, and each team's real shot total is spread
    across the game in proportion to league-typical shot rates for each situation."""
    hid, aid = str(g["home_id"]), str(g["away_id"])
    periods = gs.get("periods") or {}
    ot_len = int(_num((periods.get("4") or {}).get("length"), 300)) or 300

    def gt(per, sec):
        return (per - 1) * 1200 + sec if per <= 3 else 3600 + (per - 4) * ot_len + sec

    def side(e):
        h = str(e.get("home", ""))
        if h in ("1", "0"):
            return hid if h == "1" else aid
        tid = str(e.get("team_id") or "")
        return tid if tid in (hid, aid) else None

    items = []
    for e in gs.get("goals") or []:
        per = int(_num(e.get("period_id"), 0))
        team = side(e)
        if not per or not team:
            continue
        sec = int(_num(e.get("s"), _secs(e.get("time"))))
        items.append(dict(t=gt(per, sec), per=per, kind="goal", team=team,
                          en=str(e.get("empty_net")) == "1", pp=str(e.get("power_play")) == "1",
                          ids=[_pid(e, k) for k in ("goal_scorer", "assist1_player", "assist2_player")]))
    for e in gs.get("penalties") or []:
        per = int(_num(e.get("period_id"), 0))
        team = side(e)
        if not per or not team:
            continue
        sec = int(_num(e.get("s"), _secs(e.get("time_off_formatted"))))
        items.append(dict(t=gt(per, sec), per=per, kind="pen", team=team, mins=_num(e.get("minutes"), 2),
                          cls=str(e.get("penalty_class") or "").lower(), pid=_pid(e, "player_penalized_info")))
    items.sort(key=lambda x: (x["t"], 0 if x["kind"] == "pen" else 1))

    # lineups: dressed players, starting goalies, player shots, faceoffs
    roster, pshots, starter, fo, rinfo = [], {}, {}, {}, {}
    goalie_sa = {hid: 0, aid: 0}

    def pname(p):
        nm = f"{p.get('first_name') or ''} {p.get('last_name') or ''}".strip()
        return nm or str(p.get("name") or p.get("player_id"))

    for team, key in ((hid, "home_team_lineup"), (aid, "visitor_team_lineup")):
        lu = gs.get(key) or {}
        w = att = 0
        for p in lu.get("players") or []:
            pid = str(p.get("player_id") or "")
            if pid:
                pos = str(p.get("position_str") or p.get("position") or "F").upper()
                rinfo[pid] = ("D" if pos.startswith("D") else "F", pname(p))
                roster.append((pid, team))
                pshots[pid] = pshots.get(pid, 0) + int(_num(p.get("shots_on") or p.get("shots")))
            w += int(_num(p.get("faceoff_wins")))
            att += int(_num(p.get("faceoff_attempts")))
        fo[team] = (w, att)
        for gk in lu.get("goalies") or []:
            pid = str(gk.get("player_id") or "")
            if pid:
                rinfo[pid] = ("G", pname(gk))
            if pid and (_num(gk.get("seconds")) > 0 or str(gk.get("start")) == "1"):
                roster.append((pid, team))
            if str(gk.get("start")) == "1" and pid:
                starter[team] = pid
            goalie_sa[team] += int(_num(gk.get("shots_against")))
    gks = gs.get("goalies") or {}
    for team, key in ((hid, "home"), (aid, "visitor")):
        lst = gks.get(key) or []
        if lst and not goalie_sa[team]:
            goalie_sa[team] = int(sum(_num(x.get("shots_against")) for x in lst))
        if lst and team not in starter:
            starter[team] = str(lst[0].get("player_id") or "")

    # walk the game: penalties, power-play goals, score; record every change of situation
    pens = {hid: [], aid: []}
    score = {hid: 0, aid: 0}
    plays, marks = [], [(0, "1551", 0, 0)]

    def counts():
        return {tm: max(3, 5 - len(pens[tm])) for tm in pens}

    def sit_now():
        c = counts()
        return f"1{c[aid]}{c[hid]}1"

    def release_until(t):
        while True:
            rel = sorted(r for tm in pens for r in pens[tm] if r <= t)
            if not rel:
                return
            r = rel[0]
            for tm in pens:
                if r in pens[tm]:
                    pens[tm].remove(r)
                    break
            p = 1 + min(int(r // 1200), 3)
            plays.append((r, p, "stoppage", None, None, None, None, sit_now(), None, None))
            marks.append((r, sit_now(), score[hid], score[aid]))

    n_goals = {hid: 0, aid: 0}
    n_en = {hid: 0, aid: 0}
    for it in items:
        release_until(it["t"])
        t, per, team = it["t"], it["per"], it["team"]
        other = aid if team == hid else hid
        c = counts()
        sit = sit_now()
        if it["kind"] == "goal":
            if it["en"]:
                sit = f"{0 if team == hid else 1}{c[aid]}{c[hid]}{0 if team == aid else 1}"
                n_en[team] += 1
            plays.append((t, per, "goal", team, None, None, it["ids"][0], sit, None, starter.get(other)))
            score[team] += 1
            n_goals[team] += 1
            for pid in it["ids"]:
                if pid:
                    roster.append((pid, team))
            if (it["pp"] or c[team] > c[other]) and pens[other]:
                pens[other].remove(min(pens[other]))
        else:
            code = "MIS" if ("misconduct" in it["cls"] or it["mins"] >= 10) else ("MAJ" if it["mins"] >= 5 else "MIN")
            if code != "MIS":
                rel = t + it["mins"] * 60
                same = [r for r in pens[other] if abs(r - rel) < 1]
                if same:                       # coincidental penalties cancel out
                    pens[other].remove(same[0])
                else:
                    pens[team].append(rel)
            plays.append((t, per, "penalty", team, None, None, None, sit, code, None))
        plays.append((t, per, "stoppage", None, None, None, None, sit_now(), None, None))
        marks.append((t, sit_now(), score[hid], score[aid]))
    end = 3600
    if score[hid] == score[aid] or any(p[1] >= 4 for p in plays if p[2] == "goal"):
        last_ot = [p[0] for p in plays if p[2] == "goal" and p[1] >= 4]
        end = max(last_ot) if last_ot else 3600 + ot_len
    release_until(end)
    for p in (1, 2, 3):
        plays.append((p * 1200, p, "period-end", None, None, None, None, "1551", None, None))

    # spread each team's real shots on goal over the game, by situation
    def state_for(team, sit, hs, as_):
        if team == hid:
            st_ = classify(sit, True, hs - as_)
        else:
            st_ = classify(sit, False, as_ - hs)
        return "trail" if st_ == "en" else st_

    marks.sort(key=lambda m: m[0])
    segs = []
    for i, (t0, sit, hs, as_) in enumerate(marks):
        t1 = marks[i + 1][0] if i + 1 < len(marks) else end
        if t1 > t0:
            segs.append((t0, t1, sit, hs, as_))
    for team in (hid, aid):
        other = aid if team == hid else hid
        sog = goalie_sa[other] or int(round(LG_EV + 1))
        n = max(0, sog - (n_goals[team] - n_en[team]))
        if not n or not segs:
            continue
        w = []
        for (t0, t1, sit, hs, as_) in segs:
            rate = LG_RATE["tied"] * 1.6 if t0 >= 3600 else LG_RATE[state_for(team, sit, hs, as_)]
            w.append((t1 - t0) * rate)
        cum = np.cumsum(w)
        for k in range(n):
            target = (k + 0.5) / n * cum[-1]
            j = int(np.searchsorted(cum, target))
            t0, t1, sit, hs, as_ = segs[j]
            before = cum[j - 1] if j else 0.0
            t = t0 + (target - before) / max(w[j], 1e-9) * (t1 - t0)
            per = 1 + min(int(t // 1200), 3) if t < 3600 else 4 + int((t - 3600) // ot_len)
            plays.append((int(t), per, "shot-on-goal", team, None, None, None, sit, None, starter.get(other)))
    plays.sort(key=lambda r: (r[0], r[2] == "stoppage"))
    # per-player [shots, goals, assists, power-play points] for the backtest and rivalry pages
    pstats = {pid: [sh, 0, 0, 0] for pid, sh in pshots.items()}
    for it in items:
        if it["kind"] != "goal":
            continue
        for k, pid in enumerate(it["ids"]):
            if not pid:
                continue
            v = pstats.setdefault(pid, [0, 0, 0, 0])
            v[1 if k == 0 else 2] += 1
            v[3] += bool(it["pp"])
    return dict(id=g["id"], date=g.get("date", ""), home=g["home"], away=g["away"],
                home_id=hid, away_id=aid, home_score=g.get("hs", score[hid]), away_score=g.get("as_", score[aid]),
                plays=plays, roster=list({(p, t) for p, t in roster if p and t}), fo=fo, pshots=pshots,
                sog={hid: goalie_sa[aid], aid: goalie_sa[hid]}, rinfo=rinfo, pstats=pstats)


def fetch_pbps(ids):
    store, sess = _pbp_store(), _session()
    idx = game_index()
    missing = [i for i in ids if i not in store and i in idx]

    def one(gid):
        try:
            d = ht_get(dict(feed="gc", tab="gamesummary", game_id=gid, fmt="json", lang_code="en",
                            league_id=HT_LEAGUE_ID), sess)
            gs = (d.get("GC") or {}).get("Gamesummary") or {} if isinstance(d, dict) else {}
            return gid, slim_whl_summary(gs, idx[gid]) if gs else None
        except Exception:
            return gid, None

    if missing:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for gid, g in ex.map(one, missing):
                if g is not None:
                    store[gid] = g
    return [store[i] for i in ids if i in store]


def recent_game_ids(team, n):
    """Last n finished games for a team (topped up from last season early on)."""
    try:
        done = [g for g in all_games() if g["state"] == "FINAL" and team in (g["home"], g["away"])]
    except Exception:
        return []
    return [g["id"] for g in done[-n:]]


def _secs(mmss):
    try:
        m, s = str(mmss).split(":")[:2]
        return int(m) * 60 + int(s)
    except ValueError:
        return 0


def classify(sit, is_home, diff):
    """Game state from one team's point of view, using the NHL situation code
    (away goalie, away skaters, home skaters, home goalie)."""
    s = str(sit).zfill(4)
    try:
        ag, a_sk, h_sk, hg = (int(c) for c in s[:4])
    except ValueError:
        return "tied"
    own_g, own_s, opp_g, opp_s = (hg, h_sk, ag, a_sk) if is_home else (ag, a_sk, hg, h_sk)
    if own_g == 0 or opp_g == 0:
        return "en"
    if own_s > opp_s:
        return "pp"
    if own_s < opp_s:
        return "sh"
    return "lead" if diff > 0 else ("tied" if diff == 0 else "trail")


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def build_profile(team, n_recent):
    """Everything the simulator knows about one team, from real data."""
    ids = recent_game_ids(team, n_recent)
    games = fetch_pbps(ids)
    prof = _profile_core(team, games)
    skaters, goalies = club_players(team)
    # WHL season stats only list shooting %, so fill gaps with shots from recent game sheets
    tid = prof["team_id"]
    seen, recent_shots = Counter(), Counter()
    for g in games:
        for pid, tm in g.get("roster", []):
            if tm == tid:
                seen[pid] += 1
        for pid, sh in (g.get("pshots") or {}).items():
            recent_shots[pid] += sh
    for s_ in skaters:
        if seen.get(s_["id"]) and (s_["shots"] <= 0 or s_["g"] == 0):
            s_["shots"] = int(round(recent_shots[s_["id"]] / seen[s_["id"]] * s_["gp"]))
    prof["skaters"], prof["goalies"] = skaters, goalies
    return prof


def _profile_core(team, games):
    """Team style from a list of slimmed games (rates, shot locations, special teams, lineups)."""
    toi = dict.fromkeys(STATES, 0.0)
    sf, sa = dict.fromkeys(STATES, 0.0), dict.fromkeys(STATES, 0.0)
    shots_for, shots_ag = [], []
    gf_per, ga_per = np.zeros(4), np.zeros(4)
    sf_per, sa_per = np.zeros(4), np.zeros(4)
    ppg = ppga = pens = drawn = fo = fow = hits = 0
    starters, recent_games, lineups = [], [], []
    team_id = None
    # Recency weights: newer games count more (weights average 1, so the sample size stays honest)
    wts = 0.5 ** ((len(games) - 1 - np.arange(len(games))) / HALF_LIFE) if games else np.ones(0)
    if len(wts):
        wts = wts * len(wts) / wts.sum()

    for gi, g in enumerate(games):
        wt = float(wts[gi])
        is_home = g["home"] == team
        tid = g["home_id"] if is_home else g["away_id"]
        team_id = tid
        us = them = 0
        prev_t, prev_state, starter = 0, None, None
        for (t, per, typ, owner, x, y, _shooter, sit, tcode, goalie) in g["plays"]:
            if prev_state in toi and per <= 3:
                toi[prev_state] += wt * max(0, t - prev_t)
            prev_t = t
            state = classify(sit, is_home, us - them) if per <= 3 else "ot"
            if typ in ("shot-on-goal", "goal"):
                ours = owner == tid
                if not ours and starter is None and goalie:
                    starter = goalie
                if state in toi:
                    (sf if ours else sa)[state] += wt
                (sf_per if ours else sa_per)[min(per, 4) - 1] += 1
                if x is not None and y is not None and state not in ("en", "ot"):
                    nx, ny = (x, y) if x >= 0 else (-x, -y)
                    rec = (nx, ny, math.hypot(89 - nx, ny), typ == "goal")
                    (shots_for if ours else shots_ag).append(rec)
                if typ == "goal":
                    if ours:
                        us += 1
                        gf_per[min(per, 4) - 1] += 1
                        ppg += wt * (state == "pp")
                    else:
                        them += 1
                        ga_per[min(per, 4) - 1] += 1
                        ppga += wt * (state == "sh")
            elif typ == "penalty" and tcode in ("MIN", "BEN", "MAJ"):
                if owner == tid:
                    pens += wt
                else:
                    drawn += wt
            elif typ == "faceoff":
                fo += 1
                fow += owner == tid
            elif typ == "hit" and owner == tid:
                hits += 1
            prev_state = state
        fw_, fa_ = (g.get("fo") or {}).get(tid, (0, 0))
        fow += fw_
        fo += fa_
        starters.append(starter)
        lineups.append({pid for pid, t_ in g["roster"] if t_ == tid})
        opp = g["away"] if is_home else g["home"]
        mine, theirs = (g["home_score"], g["away_score"]) if is_home else (g["away_score"], g["home_score"])
        if mine is None:
            mine, theirs = us, them
        recent_games.append(dict(date=g["date"], opp=("vs " if is_home else "@ ") + str(opp),
                                 result=f"{'W' if mine > theirs else 'L'} {mine}-{theirs}"))

    # Shrunk shot rates (per 60) for and against in every state
    r_for, r_ag = {}, {}
    for s in STATES:
        mins = toi[s] / 60
        k = SHRINK_MIN[s]
        r_for[s] = 60 * (sf[s] + LG_RATE[s] / 60 * k) / (mins + k)
        r_ag[s] = 60 * (sa[s] + LG_RATE[MIRROR[s]] / 60 * k) / (mins + k)
    reg_min = sum(toi.values()) / 60 + 1e-9
    pen60 = 60 * (pens + LG_PEN60 / 60 * 300) / (reg_min + 300)
    n = max(len(games), 1)
    return dict(
        team=team, team_id=team_id, n_games=len(games), recent_games=recent_games,
        toi=toi, sf=sf, sa=sa, r_for=r_for, r_ag=r_ag, pen60=pen60,
        shots_for=shots_for, shots_ag=shots_ag,
        gf_per=(gf_per / n).tolist(), ga_per=(ga_per / n).tolist(),
        sf_per=(sf_per / n).tolist(), sa_per=(sa_per / n).tolist(),
        pp_pct=ppg / drawn if drawn else None, pk_pct=1 - ppga / pens if pens else None,
        pens_pg=pens / n, fo_pct=fow / fo if fo else None, hits_pg=hits / n,
        sog_pg=sum(sf_per) / n, soga_pg=sum(sa_per) / n,
        gf_pg=sum(gf_per) / n, ga_pg=sum(ga_per) / n,
        skaters=[], goalies=[], starters=starters,
        recent_lineup=set().union(*lineups[-2:]) if lineups else set(),
        last_date=max((g["date"] for g in games if g.get("date")), default=""),
    )


def club_players(team):
    """Season stats for this club's skaters and goalies (copies, so they can be adjusted safely)."""
    sk, gk = team_players(team)
    return [dict(s_) for s_ in sk], [dict(g_) for g_ in gk]


def default_lineup(prof, use_lines=True):
    """Probable starting goalie and players who sit (from DailyFaceoff lines when available,
    otherwise players who weren't in the last two lineups)."""
    gids = {g["id"] for g in prof["goalies"]}
    recent = [s for s in prof["starters"][-5:] if s in gids]
    if recent:
        c = Counter(recent)
        top = max(c.values())
        goalie = next(s for s in reversed(recent) if c[s] == top)
    elif prof["goalies"]:
        goalie = max(prof["goalies"], key=lambda g: (g["gs"], g["gp"]))["id"]
    else:
        goalie = None
    tl = team_lines(prof) if use_lines and not prof.get("asof") else None
    if tl and tl["dressed"]:
        out = tuple(sorted(s["id"] for s in prof["skaters"] if s["id"] not in tl["dressed"]))
        return goalie, out
    lineup = prof["recent_lineup"]
    out = tuple(sorted(s["id"] for s in prof["skaters"] if lineup and s["id"] not in lineup))
    return goalie, out


# --------------------------------------------------------------------------
# Building the two sides for the simulation
# --------------------------------------------------------------------------
def dist_model(dists, goals):
    b = np.clip(np.digitize(dists, DIST_EDGES) - 1, 0, 5)
    n = np.bincount(b, minlength=6)
    g = np.bincount(b, weights=goals.astype(float), minlength=6)
    return (g + DIST_PRIOR * 150) / (n + 150)


def _pool(shots, model):
    if len(shots) < 20:   # not enough real data yet: generic slot-heavy pool
        rng = np.random.default_rng(7)
        x = np.clip(rng.normal(70, 12, 300), 30, 88)
        y = np.clip(rng.normal(0, 13, 300), -38, 38)
        d = np.hypot(89 - x, y)
    else:
        arr = np.array(shots, dtype=float)
        x, y, d = arr[:, 0], arr[:, 1], arr[:, 2]
    b = np.clip(np.digitize(d, DIST_EDGES) - 1, 0, 5)
    return x, y, model[b]


def _dressed(prof, out_ids):
    sk = [s for s in prof["skaters"] if s["id"] not in set(out_ids)]
    fw = sorted([s for s in sk if s["pos"] != "D"], key=lambda s: -s["toi"])[:12]
    dm = sorted([s for s in sk if s["pos"] == "D"], key=lambda s: -s["toi"])[:6]
    return fw + dm


def _gk_factor(gk):
    """How many goals this goalie allows relative to a league-average goalie (shrunk)."""
    sv = (gk["saves"] + LG_SV * 600) / (gk["sa"] + 600)
    return float(np.clip((1 - sv) / (1 - LG_SV), 0.7, 1.4))


SLOT_TOI = {("F", 1): 1080, ("F", 2): 960, ("F", 3): 840, ("F", 4): 690,
            ("D", 1): 1380, ("D", 2): 1230, ("D", 3): 1050}      # typical total ice time (s) per line slot
PP_SHARE = {1: 0.64, 2: 0.33}                                     # share of power-play time by unit


def build_side(prof, opp, goalie_id, out_ids, model, lines=None):
    dressed = _dressed(prof, out_ids)
    if not dressed:
        dressed = [dict(id=-1, name=f"{prof['team']} skater", pos="F", gp=1, g=0, a=0, shots=2, ppg=0, toi=900)]
    gp = np.array([max(s["gp"], 1) for s in dressed], float)
    shots = np.array([s["shots"] for s in dressed], float)
    spg = shots / gp + 0.3
    toi = np.array([s["toi"] for s in dressed], float) + 1
    # role: where a player sits in tonight's lines vs the ice time he has been getting
    role = np.ones(len(dressed))
    unit = np.zeros(len(dressed), int)
    if lines:
        for i, s_ in enumerate(dressed):
            slot = lines["slot"].get(s_["id"])
            if slot in SLOT_TOI:
                role[i] = float(np.clip((SLOT_TOI[slot] / max(s_["toi"], 300)) ** 0.8, 0.7, 1.4))
            unit[i] = lines["pp"].get(s_["id"], 0)
    w_ev = spg * role / (spg * role).sum()
    w_ast = (np.array([s["a"] for s in dressed], float) / gp + 0.05) * role
    if lines and unit.any():      # power play: the listed units get the shots and the points
        share = np.array([PP_SHARE.get(u, 0.03) for u in unit])
        w_pp = spg * share
        w_ast_pp = w_ast * share
        w_ast_pp /= w_ast_pp.sum()
    else:
        w_pp = spg * (toi / toi.mean()) ** 2
        w_ast_pp = None
    w_pp /= w_pp.sum()
    w_ast /= w_ast.sum()
    if w_ast_pp is None:
        w_ast_pp = w_ast
    w_pen = np.array([s.get("pim", 0) for s in dressed], float) / gp + 0.3
    w_pen /= w_pen.sum()
    goals = np.array([s["g"] for s in dressed], float)
    fin = np.clip((goals + LG_SH_PCT * 120) / (shots + 120) / LG_SH_PCT, 0.6, 1.6)

    # Lineup strength vs the lineup the team has actually been using
    base = _dressed(prof, default_lineup(prof)[1])
    q0 = sum(s["shots"] / max(s["gp"], 1) for s in base) or 1
    q1 = sum(s["shots"] / max(s["gp"], 1) for s in dressed)
    off = float(np.clip(1 + 0.5 * (q1 / q0 - 1), 0.85, 1.15)) if base else 1.0

    gk = next((g for g in prof["goalies"] if g["id"] == goalie_id), None)
    if gk:
        gk_factor = _gk_factor(gk)
        gk_name = gk["name"]
    else:
        gk_factor, gk_name = 1.0, "Unknown goalie"

    rf, ra = prof["r_for"], opp["r_ag"]
    r_ev = np.array([rf[s] * ra[MIRROR[s]] / LG_RATE[s] for s in ("lead", "tied", "trail")])
    r_pp = rf["pp"] * ra["sh"] / LG_RATE["pp"]
    r_sh = rf["sh"] * ra["pp"] / LG_RATE["sh"]

    fx, fy, fp = _pool(prof["shots_for"], model)
    ax, ay, ap = _pool(prof["shots_ag"], model)
    return dict(
        team=prof["team"], names=[s["name"] for s in dressed], pos=[s["pos"] for s in dressed],
        ids=[s["id"] for s in dressed],
        w_ev=w_ev, w_pen=w_pen, w_pp=w_pp, w_ast=w_ast, w_ast_pp=w_ast_pp, fin=fin, pp_unit=unit.tolist(),
        r_ev=r_ev * off / 3600, r_pp=r_pp * off / 3600, r_sh=r_sh * off / 3600,
        pen_rate=prof["pen60"] / 3600, gk_factor=gk_factor, gk_name=gk_name, off=off,
        loc_for=(fx, fy, fp), loc_ag=(ax, ay, ap),
    )


# --------------------------------------------------------------------------
# The simulation engine (all games run side by side with numpy)
# --------------------------------------------------------------------------
def simulate(home, away, n, playoff=False, record=400, seed=None):
    rng = np.random.default_rng(seed)
    T = [home, away]
    P = [len(home["names"]), len(away["names"])]
    goals = np.zeros((n, 2), np.int32)
    sog = np.zeros((n, 2), np.int32)
    ppg = np.zeros((n, 2), np.int32)
    ppo = np.zeros((n, 2), np.int32)
    eng = np.zeros((n, 2), np.int32)
    so_goal = np.zeros((n, 2), np.int32)
    pen = np.zeros((n, 2), np.int32)          # seconds left that team is shorthanded
    g_mat = [np.zeros((n, p), np.int16) for p in P]
    a_mat = [np.zeros((n, p), np.int16) for p in P]
    s_mat = [np.zeros((n, p), np.int16) for p in P]
    p_mat = [np.zeros((n, p), np.int16) for p in P]   # power-play points
    goals_by_per = np.zeros((2, 4))
    sog_by_min = np.zeros((2, 66))
    events = []                                # detailed log for the first `record` games

    def draw(team, is_pp):
        out = np.empty(len(is_pp), int)
        pn, tm = P[team], T[team]
        ev_i, pp_i = np.nonzero(~is_pp)[0], np.nonzero(is_pp)[0]
        if ev_i.size:
            out[ev_i] = rng.choice(pn, ev_i.size, p=tm["w_ast"])
        if pp_i.size:
            out[pp_i] = rng.choice(pn, pp_i.size, p=tm.get("w_ast_pp", tm["w_ast"]))
        return out

    def assists(team, scorers, is_pp):
        k, pn = len(scorers), P[team]
        a1 = np.full(k, -1)
        a2 = np.full(k, -1)
        if pn < 2:
            return a1, a2
        a1 = draw(team, is_pp)
        for _ in range(8):
            bad = a1 == scorers
            if not bad.any():
                break
            a1[bad] = draw(team, is_pp[bad])
        a1[(a1 == scorers) | (rng.random(k) < 0.07)] = -1
        a2 = draw(team, is_pp)
        for _ in range(8):
            bad = (a2 == scorers) | (a2 == a1)
            if not bad.any():
                break
            a2[bad] = draw(team, is_pp[bad])
        a2[(a2 == scorers) | (a2 == a1) | (a1 < 0) | (rng.random(k) < 0.22)] = -1
        return a1, a2

    def log(sims, t, team, kind, players, a1, a2, xs, ys, strength):
        for j in np.nonzero(sims < record)[0]:
            s = sims[j]
            events.append((int(s), int(t), team, kind, int(players[j]), int(a1[j]), int(a2[j]),
                           float(xs[j]), float(ys[j]), int(goals[s, 0]), int(goals[s, 1]),
                           strength[j] if isinstance(strength, np.ndarray) else strength))

    def score(team, sims, scorers, is_pp, xs, ys, t, strength, empty_net=False):
        opp = 1 - team
        goals[sims, team] += 1
        np.add.at(g_mat[team], (sims, scorers), 1)
        a1, a2 = assists(team, scorers, np.asarray(is_pp, bool))
        for a in (a1, a2):
            ok = a >= 0
            np.add.at(a_mat[team], (sims[ok], a[ok]), 1)
        if is_pp.any():
            np.add.at(p_mat[team], (sims[is_pp], scorers[is_pp]), 1)
            for a in (a1, a2):
                ok = is_pp & (a >= 0)
                np.add.at(p_mat[team], (sims[ok], a[ok]), 1)
        ppg[sims[is_pp], team] += 1
        pen[sims[is_pp], opp] = 0                      # power-play goal ends the minor
        if empty_net:
            eng[sims, team] += 1
        goals_by_per[team, min(t // 1200, 3)] += len(sims)
        log(sims, t, team, "goal", scorers, a1, a2, xs, ys, strength)

    def step(t, active, mode):
        idx = np.nonzero(active)[0]
        k = idx.size
        if k == 0:
            return
        g0, g1 = goals[idx, 0].copy(), goals[idx, 1].copy()
        shorthanded = pen[idx] > 0
        pulled = np.zeros((k, 2), bool)
        if mode == "reg" and t >= 2400:
            rem, d = 3600 - t, g0 - g1
            pulled[:, 0] = ((d == -1) & (rem <= PULL_DOWN1)) | ((d == -2) & (rem <= PULL_DOWN2))
            pulled[:, 1] = ((d == 1) & (rem <= PULL_DOWN1)) | ((d == 2) & (rem <= PULL_DOWN2))
        for team in (0, 1):
            opp, tm, om = 1 - team, T[team], T[1 - team]
            d = (g0 - g1) if team == 0 else (g1 - g0)
            own_sh, opp_sh = shorthanded[:, team], shorthanded[:, opp]
            pp = opp_sh & ~own_sh
            sh = own_sh & ~opp_sh
            si = np.where(d > 0, 0, np.where(d == 0, 1, 2))
            if mode == "reg":
                rate = np.where(pp, tm["r_pp"], np.where(sh, tm["r_sh"], tm["r_ev"][si]))
                gmult = np.where(pp, PP_GOAL_MULT, 1.0)
            elif mode == "ot3":
                rate = np.full(k, tm["r_ev"][1] * OT3_SHOT)
                gmult = np.full(k, OT3_GOAL)
            else:
                rate = np.full(k, tm["r_ev"][1])
                gmult = np.ones(k)
            own_pull, opp_pull = pulled[:, team], pulled[:, opp]
            rate = np.where(own_pull, rate * PULL_SHOT, rate)
            gmult = np.where(own_pull, gmult * PULL_GOAL, gmult)
            u = rng.random(k)
            shot = (u < rate * DT) & ~opp_pull
            en = opp_pull & (u < EN_RATE * DT)
            strength = np.where(pp, "PP", np.where(sh, "SH", np.where(own_pull, "6v5", "EV")))
            if mode == "ot3":
                strength = np.full(k, "3v3")

            j = np.nonzero(shot)[0]
            if j.size:
                sims, kk = idx[j], j.size
                use_for = rng.random(kk) < 0.5
                fx, fy, fp = tm["loc_for"]
                ax, ay, ap = om["loc_ag"]
                lf, la = rng.integers(0, len(fx), kk), rng.integers(0, len(ax), kk)
                xs = np.where(use_for, fx[lf], ax[la]) + rng.normal(0, 1.5, kk)
                ys = np.where(use_for, fy[lf], ay[la]) + rng.normal(0, 1.5, kk)
                base = np.where(use_for, fp[lf], ap[la])
                is_pp = pp[j]
                shooter = np.where(is_pp, rng.choice(P[team], kk, p=tm["w_pp"]),
                                   rng.choice(P[team], kk, p=tm["w_ev"]))
                p = base * gmult[j] * tm["fin"][shooter] * om["gk_factor"]
                goal = rng.random(kk) < np.clip(p, 0, 0.7)
                sog[sims, team] += 1
                np.add.at(s_mat[team], (sims, shooter), 1)
                sog_by_min[team, min(t // 60, 65)] += kk
                miss = ~goal
                none = np.full(kk, -1)
                log(sims[miss], t, team, "shot", shooter[miss], none[miss], none[miss],
                    xs[miss], ys[miss], strength[j][miss])
                if goal.any():
                    score(team, sims[goal], shooter[goal], is_pp[goal], xs[goal], ys[goal],
                          t, strength[j][goal])
            j = np.nonzero(en)[0]
            if j.size:
                sims, kk = idx[j], j.size
                scorer = rng.choice(P[team], kk, p=tm["w_ev"])
                sog[sims, team] += 1
                np.add.at(s_mat[team], (sims, scorer), 1)
                score(team, sims, scorer, np.zeros(kk, bool), rng.uniform(-20, 40, kk),
                      rng.uniform(-30, 30, kk), t, "EN", empty_net=True)

            if mode == "reg" and t < 3600 - 30:
                j = np.nonzero(rng.random(k) < tm["pen_rate"] * DT)[0]
                if j.size:
                    sims = idx[j]
                    new = pen[sims, team] == 0
                    ppo[sims[new], opp] += 1
                    pen[sims, team] = np.minimum(pen[sims, team] + 120, 240)
                    pl = rng.choice(P[team], j.size, p=tm["w_pen"])
                    none = np.full(j.size, -1)
                    log(sims, t, team, "pen", pl, none, none, np.zeros(j.size), np.zeros(j.size), "")
        pen[idx] = np.maximum(pen[idx] - DT, 0)

    everyone = np.ones(n, bool)
    for s in range(3600 // DT):
        step(s * DT, everyone, "reg")
    pen[:] = 0
    tied_after_reg = goals[:, 0] == goals[:, 1]

    if not playoff:
        for s in range(300 // DT):
            step(3600 + s * DT, goals[:, 0] == goals[:, 1], "ot3")
        still = goals[:, 0] == goals[:, 1]
        p_home = float(np.clip(0.5 + 0.15 * (away["gk_factor"] - home["gk_factor"]), 0.35, 0.65))
        home_wins = rng.random(n) < p_home
        for team, mask in ((0, still & home_wins), (1, still & ~home_wins)):
            goals[mask, team] += 1
            so_goal[mask, team] += 1
            sims = np.nonzero(mask)[0]
            for s in sims[sims < record]:
                events.append((int(s), 3900, team, "so", -1, -1, -1, 0.0, 0.0,
                               int(goals[s, 0]), int(goals[s, 1]), "SO"))
        so_dec = still
    else:
        for per in range(8):
            for s in range(1200 // DT):
                step(3600 + per * 1200 + s * DT, goals[:, 0] == goals[:, 1], "ot5")
        still = goals[:, 0] == goals[:, 1]
        coin = rng.random(n) < 0.5
        goals[still & coin, 0] += 1
        goals[still & ~coin, 1] += 1
        so_dec = np.zeros(n, bool)

    return dict(goals=goals, sog=sog, ppg=ppg, ppo=ppo, eng=eng, so_goal=so_goal,
                ot_dec=tied_after_reg & ~so_dec, so_dec=so_dec, reg_dec=~tied_after_reg,
                g_mat=g_mat, a_mat=a_mat, s_mat=s_mat, p_mat=p_mat, goals_by_per=goals_by_per / n,
                sog_by_min=sog_by_min / n, events=events, n=n, record=min(record, n))


# --------------------------------------------------------------------------
# Fast team-level engine (used for the season, playoffs and win probability).
# Same rules as the detailed engine, but each simulated row can be a different
# matchup and start from any clock time and score. Only goals are tracked.
# --------------------------------------------------------------------------
LG3 = np.array([LG_RATE["lead"], LG_RATE["tied"], LG_RATE["trail"]])


def team_vector(prof, model, goalie_id=None):
    """One team's real-data profile boiled down to the numbers the fast engine needs."""
    _, out = default_lineup(prof, use_lines=False)
    dressed = _dressed(prof, out)
    if dressed:
        gp = np.array([max(s["gp"], 1) for s in dressed], float)
        shots = np.array([s["shots"] for s in dressed], float)
        goals = np.array([s["g"] for s in dressed], float)
        toi = np.array([s["toi"] for s in dressed], float) + 1
        spg = shots / gp + 0.3
        w_ev = spg / spg.sum()
        w_pp = spg * (toi / toi.mean()) ** 2
        w_pp /= w_pp.sum()
        fin = np.clip((goals + LG_SH_PCT * 120) / (shots + 120) / LG_SH_PCT, 0.6, 1.6)
        fin_ev, fin_pp = float(w_ev @ fin), float(w_pp @ fin)
    else:
        fin_ev = fin_pp = 1.0
    gks = prof["goalies"]
    gsel = next((g for g in gks if g["id"] == goalie_id), None) if goalie_id is not None else None
    if gsel:
        gk = _gk_factor(gsel)
    elif gks:   # season-long: goalies split starts the way they have been
        w = np.array([g["gs"] + 0.5 for g in gks], float)
        gk = float(np.dot(w, [_gk_factor(g) for g in gks]) / w.sum())
    else:
        gk = 1.0
    return dict(F=np.array([prof["r_for"][s] for s in STATES]), Ag=np.array([prof["r_ag"][s] for s in STATES]),
                pen=prof["pen60"] / 3600, qf=float(_pool(prof["shots_for"], model)[2].mean()),
                qa=float(_pool(prof["shots_ag"], model)[2].mean()), fin=fin_ev, fin_pp=fin_pp, gk=gk)


def stack(vecs, idx):
    return {k: np.array([v[k] for v in vecs])[idx] for k in vecs[0]}


def pair_sides(H, A, edge=HOME_EDGE):
    """Per-row rates for home team H vs away team A (both dicts of stacked arrays)."""
    def one(o, d, mult):
        qb = 0.5 * o["qf"] + 0.5 * d["qa"]
        return dict(
            r_ev=o["F"][:, :3] * d["Ag"][:, [2, 1, 0]] / LG3 * mult / 3600,
            r_pp=o["F"][:, 3] * d["Ag"][:, 4] / LG_RATE["pp"] * mult / 3600,
            r_sh=o["F"][:, 4] * d["Ag"][:, 3] / LG_RATE["sh"] * mult / 3600,
            q=qb * o["fin"] * d["gk"], q_pp=qb * o["fin_pp"] * d["gk"] * PP_GOAL_MULT,
            pen=o["pen"], gk=o["gk"])
    return [one(H, A, 1 + edge), one(A, H, 1 - edge)]


def sides_from_main(H, A, rows):
    """Fast-engine rows built from the detailed engine's two sides (same lineups/goalies)."""
    def one(o, d):
        qb = 0.5 * o["loc_for"][2].mean() + 0.5 * d["loc_ag"][2].mean()
        one_row = dict(r_ev=o["r_ev"][None, :], r_pp=np.array([o["r_pp"]]), r_sh=np.array([o["r_sh"]]),
                       q=np.array([qb * (o["w_ev"] @ o["fin"]) * d["gk_factor"]]),
                       q_pp=np.array([qb * (o["w_pp"] @ o["fin"]) * d["gk_factor"] * PP_GOAL_MULT]),
                       pen=np.array([o["pen_rate"]]), gk=np.array([o["gk_factor"]]))
        return {k: np.repeat(v, rows, axis=0) for k, v in one_row.items()}
    return [one(H, A), one(A, H)]


def fast_reg(S, start_t=None, start_g=None, seed=None):
    """Play regulation for every row; returns final goals (rows, 2)."""
    rng = np.random.default_rng(seed)
    n = len(S[0]["q"])
    st_t = np.zeros(n, int) if start_t is None else np.asarray(start_t, int)
    order = np.argsort(st_t, kind="stable")
    st_s = st_t[order]
    Ss = [{k: v[order] for k, v in side.items()} for side in S]
    goals = (np.zeros((n, 2), np.int32) if start_g is None
             else np.asarray(start_g, np.int32)[order].copy())
    pen = np.zeros((n, 2), np.int32)
    ar = np.arange(n)
    for t in range(0, 3600, DT):
        k = int(np.searchsorted(st_s, t, side="right"))
        if k == 0:
            continue
        g, pn = goals[:k], pen[:k]
        d0 = g[:, 0] - g[:, 1]
        shd = pn > 0
        pulls = None
        if t >= 2400:
            rem = 3600 - t
            pulls = (((d0 == -1) & (rem <= PULL_DOWN1)) | ((d0 == -2) & (rem <= PULL_DOWN2)),
                     ((d0 == 1) & (rem <= PULL_DOWN1)) | ((d0 == 2) & (rem <= PULL_DOWN2)))
        for team in (0, 1):
            opp, tm = 1 - team, Ss[team]
            d = d0 if team == 0 else -d0
            pp = shd[:, opp] & ~shd[:, team]
            sh = shd[:, team] & ~shd[:, opp]
            si = np.where(d > 0, 0, np.where(d == 0, 1, 2))
            rate = np.where(pp, tm["r_pp"][:k], np.where(sh, tm["r_sh"][:k], tm["r_ev"][:k][ar[:k], si]))
            q = np.where(pp, tm["q_pp"][:k], tm["q"][:k])
            if pulls is not None:
                own, other = pulls[team], pulls[opp]
                rate = np.where(own, rate * PULL_SHOT, rate)
                q = np.where(own, q * PULL_GOAL, q)
                pg = np.where(other, EN_RATE * DT, rate * DT * np.minimum(q, 0.7))
            else:
                pg = rate * DT * np.minimum(q, 0.7)
            sc = rng.random(k) < pg
            g[sc, team] += 1
            pn[sc & pp, opp] = 0
            if t < 3570:
                take = rng.random(k) < tm["pen"][:k] * DT
                pn[take, team] = np.minimum(pn[take, team] + 120, 240)
        np.maximum(pn - DT, 0, out=pn)
    inv = np.empty(n, int)
    inv[order] = np.arange(n)
    return goals[inv]


def ot_home_prob(S, playoff, rem=300.0):
    """Chance the home team wins a game that is tied going into (the rest of) overtime."""
    lam = []
    for tm in S:
        base = tm["r_ev"][:, 1] * np.minimum(tm["q"], 0.7)
        lam.append(base if playoff else base * OT3_SHOT * OT3_GOAL)
    tot = lam[0] + lam[1] + 1e-12
    share = lam[0] / tot
    if playoff:
        return share
    p_goal = 1 - np.exp(-tot * rem)
    p_so = np.clip(0.5 + 0.15 * (S[1]["gk"] - S[0]["gk"]), 0.35, 0.65)
    return share * p_goal + (1 - p_goal) * p_so


def win_prob_curve(H, A, feat, playoff, reps=1500, seed=5):
    """Home win probability through the featured game, re-simulated from each moment."""
    ev = feat["events"]
    goals_ev = [e for e in ev if e[3] == "goal"]
    end = max([e[1] for e in ev] + [3600])
    cps = set(range(0, 3600, 60)) | {e[1] + DT for e in goals_ev if e[1] + DT < 3600}
    if end > 3600:
        cps |= set(range(3600, end, 30))
    cps = sorted(cps)

    def score_at(t):
        last = (0, 0)
        for e in goals_ev:
            if e[1] < t:
                last = (e[9], e[10])
        return last

    scores = [score_at(t) for t in cps]
    reg = [i for i, t in enumerate(cps) if t < 3600]
    probs = np.zeros(len(cps))
    S = sides_from_main(H, A, len(reg) * reps)
    g = fast_reg(S, np.repeat([cps[i] for i in reg], reps),
                 np.repeat(np.array([scores[i] for i in reg]).reshape(-1, 2), reps, axis=0), seed=seed)
    d = g[:, 0] - g[:, 1]
    p = (d > 0) + (d == 0) * ot_home_prob(S, playoff)
    probs[reg] = p.reshape(len(reg), reps).mean(1)
    S1 = sides_from_main(H, A, 1)
    for i, t in enumerate(cps):
        if t >= 3600:
            hs, as_ = scores[i]
            if hs != as_:
                probs[i] = float(hs > as_)
            else:
                probs[i] = float(ot_home_prob(S1, playoff, rem=max(3900 - t, 1))[0])
    final = feat["final"]
    times = [t / 60 for t in cps] + [end / 60 + 0.5]
    return dict(t=times, home=list(probs) + [1.0 if final[0] > final[1] else 0.0])


def series_outcomes(p_home, p_away):
    """Best-of-7 (2-2-1-1-1). The 'hi' team has home ice; p_home = its win chance
    at home, p_away = on the road. Returns {('hi'|'lo', games): probability}."""
    venue = (1, 1, 0, 0, 1, 0, 1)
    out = {(w, n): 0.0 for w in ("hi", "lo") for n in (4, 5, 6, 7)}

    def rec(w, l, pr):
        g = w + l
        if w == 4:
            out[("hi", g)] += pr
            return
        if l == 4:
            out[("lo", g)] += pr
            return
        p = p_home if venue[g] else p_away
        rec(w + 1, l, pr * p)
        rec(w, l + 1, pr * (1 - p))

    rec(0, 0, 1.0)
    return out


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def league_model(n_recent, reps=300):
    """Win/OT/loss chances for every possible home-away pairing of the 32 teams."""
    teams = load_standings()["team"].tolist()
    ids = sorted({i for t in teams for i in recent_game_ids(t, n_recent)})
    fetch_pbps(ids)
    profs = {t: build_profile(t, n_recent) for t in teams}
    allshots = [s for p in profs.values() for s in p["shots_for"]]
    if allshots:
        arr = np.array(allshots, dtype=float)
        model = dist_model(arr[:, 2], arr[:, 3])
    else:
        model = DIST_PRIOR.copy()
    vecs = [team_vector(profs[t], model) for t in teams]
    N = len(teams)
    hi, ai = np.array([(i, j) for i in range(N) for j in range(N) if i != j]).T
    S = pair_sides(stack(vecs, np.repeat(hi, reps)), stack(vecs, np.repeat(ai, reps)))
    g = fast_reg(S, seed=11)
    d = g[:, 0] - g[:, 1]
    tie = d == 0
    ph_reg, ph_po = ot_home_prob(S, False), ot_home_prob(S, True)

    def agg(x):
        return np.asarray(x, float).reshape(-1, reps).mean(1)

    R = np.zeros((N, N, 4))       # home reg win, home OT/SO win, away OT/SO win, away reg win
    R[hi, ai, 0] = agg(d > 0)
    R[hi, ai, 1] = agg(tie * ph_reg)
    R[hi, ai, 2] = agg(tie * (1 - ph_reg))
    R[hi, ai, 3] = agg(d < 0)
    W = np.zeros((N, N))          # playoff game: P(home team wins)
    W[hi, ai] = agg((d > 0) + tie * ph_po)
    SP = np.zeros((N, N))         # P(row team wins a series with home ice)
    for i in range(N):
        for j in range(N):
            if i != j:
                o = series_outcomes(W[i, j], 1 - W[j, i])
                SP[i, j] = sum(v for (w, _), v in o.items() if w == "hi")
    return dict(teams=teams, R=R, W=W, SP=SP)


def remaining_games():
    try:
        return [(g["home"], g["away"], g["date"]) for g in season_schedule(season_id()) if g["state"] != "FINAL"]
    except Exception:
        return []


def make_bracket(key, div_members):
    """WHL format: top 8 per conference, division winners seeded 1-2, then by points.
    Returns {conf: [8 team indexes in seed order]}."""
    out = {}
    for conf, divs in CONF_DIVS.items():
        teams = [t for d in divs for t in div_members.get(d, [])]
        if len(teams) < 8:
            continue
        winners = sorted([max(div_members[d], key=lambda t: key[t]) for d in divs if div_members.get(d)],
                         key=lambda t: -key[t])
        rest = sorted([t for t in teams if t not in winners], key=lambda t: -key[t])
        out[conf] = (winners + rest)[:8]
    return out


def play_bracket(br, key, SP, rng, counts):
    """Play out a bracket once (re-seeding after round 1); counts[team, round] += 1 for
    rounds reached (0 playoffs, 1 second round, 2 conference final, 3 final, 4 champion)."""
    def series(a, b):
        hi, lo = (a, b) if key[a] >= key[b] else (b, a)
        return hi if rng.random() < SP[hi, lo] else lo

    champs = []
    for seeds in br.values():
        rank = {t: i for i, t in enumerate(seeds)}
        for t in seeds:
            counts[t, 0] += 1
        r1 = [series(seeds[i], seeds[7 - i]) for i in range(4)]
        for t in r1:
            counts[t, 1] += 1
        r1.sort(key=lambda t: rank[t])
        r2 = [series(r1[0], r1[3]), series(r1[1], r1[2])]
        for t in r2:
            counts[t, 2] += 1
        cw = series(*r2)
        counts[cw, 3] += 1
        champs.append(cw)
    if len(champs) == 2:
        counts[series(*champs), 4] += 1


@st.cache_data(ttl=3600, show_spinner=False, max_entries=6)
def season_sim(n_recent, n_seasons, seed=None):
    lm = league_model(n_recent)
    teams, SP = lm["teams"], lm["SP"]
    N, idx = len(teams), {t: i for i, t in enumerate(teams)}
    stn = load_standings().set_index("team").loc[teams]
    rng = np.random.default_rng(seed)
    rem = [(idx[h], idx[a]) for h, a, _ in remaining_games() if h in idx and a in idx]
    pts0, rw0 = stn["pts"].to_numpy(float), stn["rw"].to_numpy(float)
    pts = np.tile(pts0, (n_seasons, 1))
    rw = np.tile(rw0, (n_seasons, 1))
    if rem:
        h, a = np.array(rem).T
        G = len(h)
        cum = lm["R"][h, a].cumsum(1)[:, :3]
        Hm = np.zeros((G, N), np.float32)
        Hm[np.arange(G), h] = 1
        Am = np.zeros((G, N), np.float32)
        Am[np.arange(G), a] = 1
        HP, AP = np.array([2, 2, 1, 0], np.float32), np.array([0, 1, 2, 2], np.float32)
        HR, AR = np.array([1, 0, 0, 0], np.float32), np.array([0, 0, 0, 1], np.float32)
        for s0 in range(0, n_seasons, 1000):
            s1 = min(s0 + 1000, n_seasons)
            u = rng.random((s1 - s0, G))
            oc = (u[:, :, None] > cum[None]).sum(2)
            pts[s0:s1] += HP[oc] @ Hm + AP[oc] @ Am
            rw[s0:s1] += HR[oc] @ Hm + AR[oc] @ Am
    key = pts + rw * 1e-3 + rng.random(pts.shape) * 1e-6
    divs = stn["div"].tolist()
    div_members = {d: [i for i in range(N) if divs[i] == d] for d in CONF_OF}
    div_win = np.zeros(N)
    for d, mem in div_members.items():
        if mem:
            w = np.array(mem)[key[:, mem].argmax(1)]
            div_win += np.bincount(w, minlength=N)
    pres = np.bincount(key.argmax(1), minlength=N)
    counts = np.zeros((N, 5))
    for s in range(n_seasons):
        play_bracket(make_bracket(key[s], div_members), key[s], SP, rng, counts)
    df = pd.DataFrame({
        "Team": teams, "Div": divs, "GP": stn["gp"].to_numpy(), "PTS now": pts0.astype(int),
        "Proj PTS": pts.mean(0), "Low (10%)": np.percentile(pts, 10, axis=0),
        "High (90%)": np.percentile(pts, 90, axis=0),
        "Playoffs %": counts[:, 0] / n_seasons * 100, "Division %": div_win / n_seasons * 100,
        "Best record %": pres / n_seasons * 100, "Conf final %": counts[:, 2] / n_seasons * 100,
        "Final %": counts[:, 3] / n_seasons * 100, "Cup %": counts[:, 4] / n_seasons * 100,
    })
    return dict(table=df.sort_values("Proj PTS", ascending=False).reset_index(drop=True),
                n_games=len(rem), n_seasons=n_seasons)


@st.cache_data(ttl=1800, show_spinner=False)
def today_bracket(n_recent, n=20000):
    """Bracket from the current standings, played out n times."""
    lm = league_model(n_recent)
    teams, SP = lm["teams"], lm["SP"]
    N = len(teams)
    stn = load_standings().set_index("team").loc[teams]
    key = stn["pts"].to_numpy(float) + stn["rw"].to_numpy(float) * 1e-3 - stn["gp"].to_numpy(float) * 1e-6
    divs = stn["div"].tolist()
    br = make_bracket(key, {d: [i for i in range(N) if divs[i] == d] for d in CONF_OF})
    counts = np.zeros((N, 5))
    rng = np.random.default_rng(1)
    if br:
        for _ in range(n):
            play_bracket(br, key, SP, rng, counts)
    return dict(teams=teams, SP=SP, key=key, br=br, counts=counts, n=n)


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def pair_series(a, b, n_recent, reps=8000):
    """Per-game playoff win chances for a vs b in both buildings (a = row 0 of output)."""
    pa, pb = build_profile(a, n_recent), build_profile(b, n_recent)
    shots = pa["shots_for"] + pb["shots_for"] + pa["shots_ag"] + pb["shots_ag"]
    model = dist_model(np.array(shots, float)[:, 2], np.array(shots, float)[:, 3]) if shots else DIST_PRIOR.copy()
    va, vb = team_vector(pa, model), team_vector(pb, model)
    out = []
    for home, away in ((va, vb), (vb, va)):
        S = pair_sides(stack([home], np.zeros(reps, int)), stack([away], np.zeros(reps, int)))
        g = fast_reg(S, seed=3)
        d = g[:, 0] - g[:, 1]
        out.append(float(((d > 0) + (d == 0) * ot_home_prob(S, True)).mean()))
    return out[0], 1 - out[1]      # a wins at home, a wins on the road


@st.cache_data(ttl=90, show_spinner=False)
def live_prob(home, away, elapsed, hs, as_, n_recent, playoff=False, reps=4000):
    """Home win chance for a game in progress: replay the rest of it from the real score and clock."""
    hp_, ap_ = build_profile(home, n_recent), build_profile(away, n_recent)
    shots = hp_["shots_for"] + hp_["shots_ag"] + ap_["shots_for"] + ap_["shots_ag"]
    model = dist_model(np.array(shots, float)[:, 2], np.array(shots, float)[:, 3]) if shots else DIST_PRIOR.copy()
    S = pair_sides(stack([team_vector(hp_, model)], np.zeros(reps, int)),
                   stack([team_vector(ap_, model)], np.zeros(reps, int)))
    if elapsed >= 3600:          # overtime
        if hs != as_:
            return float(hs > as_)
        rem = (1200 if playoff else 300) - (elapsed - 3600)
        return float(ot_home_prob(S, playoff, rem=max(rem, 1) if not playoff else 300.0).mean())
    t0 = int(elapsed) // DT * DT
    g = fast_reg(S, np.full(reps, t0), np.tile([hs, as_], (reps, 1)), seed=17)
    d = g[:, 0] - g[:, 1]
    return float(((d > 0) + (d == 0) * ot_home_prob(S, playoff)).mean())


# --------------------------------------------------------------------------
# Running a matchup + summarising
# --------------------------------------------------------------------------
def is_b2b(prof, game_date):
    """True if this team also played the day before game_date (second night of a back-to-back)."""
    try:
        return bool(prof["last_date"]) and (dt.date.fromisoformat(str(game_date)[:10])
                                             - dt.date.fromisoformat(prof["last_date"][:10])).days == 1
    except Exception:
        return False


@st.cache_data(show_spinner=False, max_entries=60)
def run_matchup(home, away, h_goalie, a_goalie, h_out, a_out, n, playoff, n_recent, seed=None, game_date=None,
                use_lines=True, energy=None, home_ice=True):
    """energy = (home level, away level) from FATIGUE_LEVELS to override the schedule (what-if lab)."""
    hp, ap = build_profile(home, n_recent), build_profile(away, n_recent)
    shots = hp["shots_for"] + hp["shots_ag"] + ap["shots_for"] + ap["shots_ag"]
    if shots:
        arr = np.array(shots, dtype=float)
        model = dist_model(arr[:, 2], arr[:, 3])
    else:
        model = DIST_PRIOR.copy()
    hl = team_lines(hp) if use_lines else None
    al = team_lines(ap) if use_lines else None
    H = build_side(hp, ap, h_goalie, h_out, model, hl)
    A = build_side(ap, hp, a_goalie, a_out, model, al)
    edge = HOME_EDGE if home_ice else 0.0
    for side, mult in ((H, 1 + edge), (A, 1 - edge)):
        for key in ("r_ev", "r_pp", "r_sh"):
            side[key] = side[key] * mult
    fat = dict(home=fatigue_info(home, game_date, home) if game_date else None,
               away=fatigue_info(away, game_date, home) if game_date else None)
    if energy:
        f = {"home": FATIGUE_LEVELS[energy[0]], "away": FATIGUE_LEVELS[energy[1]]}
    else:
        f = {k: ((v["f_for"], v["f_ag"]) if v else (1.0, 1.0)) for k, v in fat.items()}
    for key in ("r_ev", "r_pp", "r_sh"):
        H[key] = H[key] * f["home"][0] * f["away"][1]
        A[key] = A[key] * f["away"][0] * f["home"][1]
    b2b = {k: bool(v and v["b2b"]) for k, v in fat.items()}
    r = simulate(H, A, n, playoff=playoff, seed=seed)
    res = summarize(r, H, A)
    res["winprob"] = win_prob_curve(H, A, res["featured"], playoff)
    res["playoff"] = playoff
    res["b2b"] = b2b
    res["fatigue"] = fat
    res["game_date"] = game_date
    res["lines"] = dict(home=(hl or {}).get("updated"), away=(al or {}).get("updated"))
    res["props"][0]["goalie_id"], res["props"][1]["goalie_id"] = h_goalie, a_goalie
    return res


def summarize(r, H, A):
    g, n = r["goals"], r["n"]
    hw = g[:, 0] > g[:, 1]
    scores = Counter(map(tuple, g.tolist()))
    top = scores.most_common(12)

    def players(team, side):
        gm, am, sm = r["g_mat"][team], r["a_mat"][team], r["s_mat"][team]
        pts = gm + am
        df = pd.DataFrame(dict(
            Player=side["names"], Pos=side["pos"],
            G=gm.mean(0), A=am.mean(0), PTS=pts.mean(0), SOG=sm.mean(0),
            **{"Goal %": (gm > 0).mean(0) * 100, "Point %": (pts > 0).mean(0) * 100},
        ))
        return df.sort_values("PTS", ascending=False).reset_index(drop=True)

    def goalie(team, side):
        opp = 1 - team
        sa = r["sog"][:, opp] - r["eng"][:, opp]
        ga = g[:, opp] - r["eng"][:, opp] - r["so_goal"][:, opp]
        won = hw if team == 0 else ~hw
        return dict(Goalie=side["gk_name"], Saves=(sa - ga).mean(), GA=ga.mean(),
                    **{"SV%": (sa - ga).sum() / max(sa.sum(), 1),
                       "Shutout %": ((g[:, opp] - r["so_goal"][:, opp]) == 0).mean() * 100,
                       "Win %": won.mean() * 100})

    # Featured game: one of the recorded games with the most common final score
    modal = top[0][0]
    rec = r["record"]
    cands = [i for i in range(rec) if tuple(g[i]) == modal]
    if not cands:
        cands = [int(np.argmin(np.abs(g[:rec, 0] - modal[0]) + np.abs(g[:rec, 1] - modal[1])))]
    feat = cands[0]
    ev = sorted([e for e in r["events"] if e[0] == feat], key=lambda e: (e[1], e[3] != "pen"))

    return dict(
        home=H["team"], away=A["team"], n=n,
        home_win=hw.mean(), away_win=1 - hw.mean(),
        reg=r["reg_dec"].mean(), ot=r["ot_dec"].mean(), so=r["so_dec"].mean(),
        home_reg_win=(hw & r["reg_dec"]).mean(), away_reg_win=(~hw & r["reg_dec"]).mean(),
        goals_mean=g.mean(0), sog_mean=r["sog"].mean(0), ppg_mean=r["ppg"].mean(0),
        ppo_mean=r["ppo"].mean(0), eng_mean=r["eng"].mean(0),
        home_cover=((g[:, 0] - g[:, 1]) >= 2).mean(), away_cover=((g[:, 1] - g[:, 0]) >= 2).mean(),
        total=g.sum(1).astype(np.int16), margin=(g[:, 0] - g[:, 1]).astype(np.int16), top_scores=[(s, c / n) for s, c in top],
        goal_dist=[np.bincount(np.minimum(g[:, i], 9), minlength=10) / n for i in (0, 1)],
        players=[players(0, H), players(1, A)], goalies=[goalie(0, H), goalie(1, A)],
        goals_by_per=r["goals_by_per"], sog_by_min=r["sog_by_min"],
        featured=dict(events=ev, final=tuple(int(v) for v in g[feat]),
                      ot=bool(r["ot_dec"][feat]), so=bool(r["so_dec"][feat]),
                      names=[H["names"], A["names"]]),
        sides=dict(home=dict(off=H["off"], gk=H["gk_name"], gkf=H["gk_factor"]),
                   away=dict(off=A["off"], gk=A["gk_name"], gkf=A["gk_factor"])),
        props=[prop_block(r, 0, H), prop_block(r, 1, A)],
    )


def prop_block(r, team, side):
    """Per-simulation stat lines for every dressed player (used for player props)."""
    opp = 1 - team
    sa = r["sog"][:, opp] - r["eng"][:, opp]
    ga = r["goals"][:, opp] - r["eng"][:, opp] - r["so_goal"][:, opp]
    c8 = lambda m: np.minimum(m, 127).astype(np.int8)
    return dict(team=side["team"], ids=list(side["ids"]), names=list(side["names"]), pos=list(side["pos"]),
                G=c8(r["g_mat"][team]), A=c8(r["a_mat"][team]), SOG=c8(r["s_mat"][team]),
                PPP=c8(r["p_mat"][team]), saves=(sa - ga).astype(np.int16), ga=ga.astype(np.int16),
                goalie=side["gk_name"])


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def clock(t):
    per = t // 1200 + 1
    rem = 1200 - t % 1200
    if per >= 4:
        label = "OT" if per == 4 else f"{per - 3}OT"
    else:
        label = f"P{per}"
    return f"{label} {rem // 60:02d}:{rem % 60:02d}"


def rink_shapes():
    red, blue, line = "#d62728", "#1f4e9c", dict(width=2)
    corner = ("M -72,-42.5 L 72,-42.5 Q 100,-42.5 100,-14.5 L 100,14.5 Q 100,42.5 72,42.5 "
              "L -72,42.5 Q -100,42.5 -100,14.5 L -100,-14.5 Q -100,-42.5 -72,-42.5 Z")
    shapes = [dict(type="path", path=corner, line=dict(color="#333", width=3),
                   fillcolor="#f7fbff", layer="below")]
    shapes.append(dict(type="line", x0=0, x1=0, y0=-42.5, y1=42.5, line=dict(color=red, width=4), layer="below"))
    for x in (-25, 25):
        shapes.append(dict(type="line", x0=x, x1=x, y0=-42.5, y1=42.5, line=dict(color=blue, width=5), layer="below"))
    for x in (-89, 89):
        shapes.append(dict(type="line", x0=x, x1=x, y0=-36.7, y1=36.7, line=dict(color=red, **line), layer="below"))
        sgn = 1 if x > 0 else -1
        shapes.append(dict(type="rect", x0=89 * sgn, x1=92.5 * sgn, y0=-3, y1=3,
                           line=dict(color="#333", width=1.5), fillcolor="#ddd", layer="below"))
        shapes.append(dict(type="path", layer="below", fillcolor="rgba(70,130,220,0.25)",
                           line=dict(color=red, width=1),
                           path=f"M {89 * sgn},-4 L {(89 - 4.5) * sgn},-4 Q {(89 - 6.5) * sgn},0 {(89 - 4.5) * sgn},4 L {89 * sgn},4 Z"))
    shapes.append(dict(type="circle", x0=-15, x1=15, y0=-15, y1=15, line=dict(color=blue, **line), layer="below"))
    for x in (-69, 69):
        for y in (-22, 22):
            shapes.append(dict(type="circle", x0=x - 15, x1=x + 15, y0=y - 15, y1=y + 15,
                               line=dict(color=red, **line), layer="below"))
    for x in (-20, 20):
        for y in (-22, 22):
            shapes.append(dict(type="circle", x0=x - 1, x1=x + 1, y0=y - 1, y1=y + 1,
                               line=dict(color=red), fillcolor=red, layer="below"))
    return shapes


def rink_layout(height=520):
    return dict(
        shapes=rink_shapes(), height=height, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-102, 102], visible=False, fixedrange=True),
        yaxis=dict(range=[-45, 45], visible=False, scaleanchor="x", fixedrange=True),
        margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h", y=-0.02),
    )


def event_text(e, names, home, away):
    _, t, team, kind, pl, a1, a2, *_rest = e
    strength = e[11]
    ab = home if team == 0 else away
    nm = names[team]
    who = nm[pl] if 0 <= pl < len(nm) else ""
    if kind == "goal":
        ast = ", ".join(nm[a] for a in (a1, a2) if 0 <= a < len(nm)) or "unassisted"
        tag = f" ({strength})" if strength not in ("EV", "", "3v3") else (" (3v3 OT)" if strength == "3v3" else "")
        return f"🚨 {ab} GOAL{tag}: {who} ({ast})"
    if kind == "shot":
        return f"{ab} shot on goal: {who}"
    if kind == "pen":
        return f"⛔ {ab} penalty: {who} (2 min)"
    if kind == "so":
        return f"🥅 {ab} wins the shootout"
    return kind


ANIM_TEMPLATE = r"""<meta charset="utf-8">
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: transparent; font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #e8eefc; }
  #app { background: #0b1220; border-radius: 14px; overflow: hidden; max-width: 1200px; margin: 0 auto;
         user-select: none; -webkit-user-select: none; -webkit-tap-highlight-color: transparent; }
  #app:fullscreen { max-width: none; border-radius: 0; display: flex; flex-direction: column; justify-content: center; }
  #sb { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 6px; padding: 8px 12px;
        background: linear-gradient(180deg, #18233f, #0f182b); border-bottom: 1px solid #ffffff12; }
  .tm { display: flex; align-items: center; gap: 8px; min-width: 0; }
  .tm.home { justify-content: flex-end; }
  .chip { width: 6px; height: 34px; border-radius: 3px; }
  .logo { width: 38px; height: 38px; object-fit: contain; filter: drop-shadow(0 1px 2px #000a); }
  .nm { display: flex; flex-direction: column; min-width: 0; }
  .tm.home .nm { align-items: flex-end; }
  .ab { font-weight: 800; font-size: 18px; letter-spacing: .5px; line-height: 1.1; }
  .sog { font-size: 11px; color: #8ea0c4; white-space: nowrap; }
  .sc { font-weight: 900; font-size: 32px; min-width: 30px; text-align: center; font-variant-numeric: tabular-nums;
        transition: transform .2s, color .2s; }
  .sc.pop { transform: scale(1.35); color: #ffd84d; }
  #mid { text-align: center; min-width: 96px; }
  #clk { font-weight: 800; font-size: 18px; font-variant-numeric: tabular-nums; }
  #per { font-size: 10.5px; color: #8ea0c4; text-transform: uppercase; letter-spacing: 1px; }
  #pp { display: inline-block; margin-top: 3px; font-size: 11px; font-weight: 800; padding: 1px 7px; border-radius: 6px;
        background: #f5c518; color: #111; visibility: hidden; font-variant-numeric: tabular-nums; }
  #stage { position: relative; background: #0b1220; }
  canvas { display: block; width: 100%; }
  #rink { touch-action: pan-y; cursor: pointer; }
  #big { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; pointer-events: none; }
  #bigbtn { pointer-events: auto; width: 74px; height: 74px; border-radius: 50%; border: 0; background: #0b1220cc;
            color: #fff; font-size: 30px; box-shadow: 0 6px 24px #0008; cursor: pointer; }
  #banner { position: absolute; left: 50%; top: 10%; transform: translate(-50%, -16px) scale(.92); opacity: 0;
            transition: opacity .25s, transform .25s; pointer-events: none; text-align: center; padding: 10px 22px;
            border-radius: 14px; color: #fff; box-shadow: 0 10px 30px #0009; max-width: 92%; }
  #banner.show { opacity: 1; transform: translate(-50%, 0) scale(1); }
  #banner .t1 { font-weight: 900; font-size: 30px; letter-spacing: 2px; }
  #banner .t2 { font-weight: 700; font-size: 16px; margin-top: 2px; }
  #banner .t3 { font-size: 12.5px; opacity: .9; margin-top: 2px; }
  #card { position: absolute; inset: 0; display: none; align-items: center; justify-content: center; background: #0b1220c4; }
  #card.show { display: flex; }
  #card .box { background: #131d33; border: 1px solid #ffffff1a; border-radius: 16px; padding: 16px 22px; text-align: center;
               min-width: 240px; max-width: 92%; box-shadow: 0 12px 40px #000a; }
  #card h3 { margin: 0 0 6px; font-size: 18px; letter-spacing: 1px; text-transform: uppercase; color: #ffd84d; }
  #card .row { font-size: 14px; margin: 3px 0; color: #dbe4f7; }
  #card .stars { text-align: left; margin-top: 8px; }
  #card button { margin-top: 10px; }
  #tlwrap { padding: 6px 10px 2px; background: #0e1628; }
  #tl { height: 64px; cursor: pointer; touch-action: none; }
  #wpl { display: flex; justify-content: space-between; font-size: 11px; color: #8ea0c4; padding: 0 2px 2px; }
  #ctrl { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; justify-content: center; padding: 8px 10px; background: #0e1628; }
  .btn { background: #1c2742; color: #e8eefc; border: 1px solid #ffffff1a; border-radius: 9px; padding: 7px 11px;
         font-size: 14px; cursor: pointer; line-height: 1; min-height: 34px; }
  .btn:hover { background: #26345a; }
  .btn.on { background: #e8eefc; color: #0b1220; font-weight: 700; }
  .grp { display: inline-flex; border-radius: 9px; overflow: hidden; border: 1px solid #ffffff1a; }
  .grp .btn { border: 0; border-radius: 0; border-right: 1px solid #ffffff12; padding: 7px 9px; }
  .grp .btn:last-child { border-right: 0; }
  #tick { padding: 6px 12px 10px; background: #0e1628; font-size: 13px; min-height: 74px; }
  #tick div { padding: 2px 0; color: #9fb0d2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  #tick div.g { color: #fff; font-weight: 700; }
  #tick div.p { color: #f5c518; }
  #tick .tt { display: inline-block; min-width: 76px; color: #6f81a6; font-variant-numeric: tabular-nums; }
  #call { min-height: 30px; padding: 6px 12px; background: linear-gradient(90deg, #101a30, #16223d, #101a30);
          color: #ffd84d; font-weight: 800; font-size: 14px; text-align: center; border-bottom: 1px solid #ffffff10;
          transition: opacity .5s, background .3s; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  #call.dim { opacity: .45; }
  #call.goal { color: #fff; background: linear-gradient(90deg, #5a0d18, #b3162f, #5a0d18); }
  #call.pen { color: #111; background: linear-gradient(90deg, #a88a12, #f5c518, #a88a12); }
  @media (max-width: 600px) {
    #call { font-size: 12.5px; padding: 5px 8px; white-space: normal; }
    #sb { padding: 6px 8px; }
    .logo { width: 28px; height: 28px; }
    .chip { display: none; }
    .ab { font-size: 15px; }
    .sc { font-size: 26px; min-width: 22px; }
    #clk { font-size: 15px; }
    #mid { min-width: 78px; }
    .btn { padding: 8px 9px; font-size: 13px; }
    #banner { top: 6%; padding: 6px 14px; }
    #banner .t1 { font-size: 20px; }
    #banner .t2 { font-size: 13px; }
    #banner .t3 { font-size: 11px; }
    #tick { font-size: 12px; }
    #tick .tt { min-width: 64px; }
  }
</style>
<div id="app" tabindex="0">
  <div id="sb">
    <div class="tm away"><span class="chip" id="ach"></span><img class="logo" id="alogo" alt=""><div class="nm"><span class="ab" id="aab"></span><span class="sog" id="asog"></span></div><span class="sc" id="asc">0</span></div>
    <div id="mid"><div id="clk">20:00</div><div id="per">1st period</div><span id="pp"></span></div>
    <div class="tm home"><span class="sc" id="hsc">0</span><div class="nm"><span class="ab" id="hab"></span><span class="sog" id="hsog"></span></div><img class="logo" id="hlogo" alt=""><span class="chip" id="hch"></span></div>
  </div>
  <div id="call">🎙️ Welcome to the rink! Press ▶ for puck drop.</div>
  <div id="stage">
    <canvas id="rink"></canvas>
    <div id="big"><button id="bigbtn" aria-label="Play">▶</button></div>
    <div id="banner"><div class="t1">GOAL!</div><div class="t2"></div><div class="t3"></div></div>
    <div id="card"><div class="box" id="cardbox"></div></div>
  </div>
  <div id="tlwrap"><div id="wpl"><span id="wpa"></span><span id="wpm">Win probability · tap or drag to jump</span><span id="wph"></span></div><canvas id="tl"></canvas></div>
  <div id="ctrl">
    <button class="btn" id="b_restart" title="Restart">⏮</button>
    <button class="btn" id="b_prev" title="Previous goal (←)">⏪ Goal</button>
    <button class="btn" id="b_play" title="Play / pause (space)" style="min-width:74px">▶ Play</button>
    <button class="btn" id="b_next" title="Next goal (→)">Goal ⏩</button>
    <span class="grp" id="speeds"></span>
    <button class="btn" id="b_cam" title="Camera">🎥 Follow</button>
    <button class="btn" id="b_snd" title="Goal horn">🔇</button>
    <button class="btn" id="b_fs" title="Full screen">⛶</button>
  </div>
  <div id="tick"></div>
</div>
<script>
(function () {
const D = __DATA__;
const $ = (id) => document.getElementById(id);
const COL = [D.hc, D.ac];                       // 0 = home, 1 = away
const AB = [D.home, D.away];
const EV = D.events.slice().sort((a, b) => a.t - b.t);
const OT3 = !D.playoff;
const SO = EV.find((e) => e.kind === "so");
const END = D.end;
const GOALS = EV.filter((e) => e.kind === "goal");
const PENS = EV.filter((e) => e.kind === "pen");
const SHOTS = EV.filter((e) => e.kind === "shot" || e.kind === "goal");
const DIR = (team) => (team === 0 ? 1 : -1);    // home attacks +x, away attacks -x
const NUM = D.nums || [{}, {}], LINES = D.lines || [{ F: [], D: [] }, { F: [], D: [] }], GNUM = D.gnums || ["", ""];
const jn = (team, name) => (NUM[team] && NUM[team][name] ? "#" + NUM[team][name] + " " : "");
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const R = mulberry32(20261115);
const rnd = (a, b) => a + (b - a) * R();

function onIce(x, y) {          // keep a point inside the rounded boards
  x = clamp(x, -94, 94); y = clamp(y, -38, 38);
  const ax = Math.abs(x);
  if (ax > 72) { const lim = 38 - (ax - 72) * 0.9; y = clamp(y, -Math.max(lim, 6), Math.max(lim, 6)); }
  return [x, y];
}

// ---------- puck path (keyframes) ----------
const K = [];
function kf(t, x, y, own, opt) {
  opt = opt || {};
  if (K.length && t <= K[K.length - 1].t + 0.04) t = K[K.length - 1].t + 0.05;
  const p = opt.raw ? [x, y] : onIce(x, y);
  K.push({ t, x: p[0], y: p[1], own, jump: !!opt.jump, straight: !!opt.straight });
}
function fill(t1, team, tx, ty) {               // carry the puck around until t1, ending near (tx, ty)
  const last = K[K.length - 1];
  const dur = t1 - last.t;
  if (dur <= 1.2) return;
  const n = Math.floor(dur / 3.2);
  let own = last.own >= 0 ? last.own : (R() < 0.5 ? 0 : 1);
  for (let i = 1; i <= n; i++) {
    const t = last.t + dur * i / (n + 1);
    const late = team >= 0 && (n - i) < 2;      // the last two touches belong to the shooting team
    if (late) own = team; else if (R() < 0.33) own = 1 - own;
    const d = DIR(own);
    let x, y;
    if (late) { x = d * rnd(38, 72); y = rnd(-30, 30); if (i === n && tx !== null) { x = (x + tx) / 2; y = (y + ty) / 2 + rnd(-8, 8); } }
    else { x = d * rnd(-35, 78); y = rnd(-35, 35); }
    kf(t, x, y, own);
  }
}
const breaks = [1200, 2400];
if (END > 3600 || SO) breaks.push(3600);
if (!OT3) for (let b = 4800; b < END; b += 1200) breaks.push(b);
const items = EV.map((e) => ({ t: e.t, e })).concat(breaks.map((b) => ({ t: b, brk: true })));
items.sort((a, b) => a.t - b.t || (a.brk ? -1 : 1));
kf(0, 0, 0, -1, { jump: true });
for (const it of items) {
  if (it.brk) { fill(it.t - 1, -1, null, null); kf(it.t, 0, 0, -1, { jump: true }); continue; }
  const e = it.e, d = DIR(e.team);
  if (e.kind === "shot" || e.kind === "goal") {
    const ex = e.x, ey = e.y;
    fill(e.t - 1.1, e.team, ex, ey);
    kf(e.t - 1.0, ex, ey, e.team);
    if (e.kind === "goal") {
      kf(e.t, d * 91.2, clamp(ey * 0.06, -1.8, 1.8), e.team, { straight: true, raw: true });
      kf(e.t + 0.6, 0, 0, -1, { jump: true });
    } else {
      kf(e.t, d * 86.6, clamp(ey * 0.08, -2.4, 2.4), e.team, { straight: true, raw: true });
      if (R() < 0.55) kf(e.t + 0.9, d * rnd(72, 84), (R() < 0.5 ? -1 : 1) * rnd(10, 28), -1);
      else kf(e.t + 1.4, d * 69, ey >= 0 ? 22 : -22, -1, { jump: true });
    }
  } else if (e.kind === "pen") {
    fill(e.t, -1, null, null);
    kf(e.t + 0.4, -d * 69, R() < 0.5 ? 22 : -22, -1, { jump: true });
  } else if (e.kind === "so") {
    fill(Math.min(e.t, END) - 1, -1, null, null);
    kf(e.t, 0, 0, -1, { jump: true });
  }
}
fill(END, -1, null, null);
kf(END + 0.5, K[K.length - 1].x, K[K.length - 1].y, -1);
const KT = K.map((k) => k.t);

function seg(T) {           // index i with K[i].t <= T < K[i+1].t
  let lo = 0, hi = K.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (KT[m] <= T) lo = m; else hi = m; }
  return lo;
}
function puckAt(T) {
  const i = seg(T), a = K[i], b = K[Math.min(i + 1, K.length - 1)];
  if (b === a || b.jump || T < a.t) return [a.x, a.y, a.own, 0];
  let f = clamp((T - a.t) / (b.t - a.t), 0, 1);
  if (b.straight) return [a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f, b.own, 1];
  const p0 = (i > 0 && !a.jump) ? K[i - 1] : a;
  const c = K[i + 2];
  const p3 = (c && !c.jump) ? c : b;
  const f2 = f * f, f3 = f2 * f;
  const cr = (q0, q1, q2, q3) => 0.5 * (2 * q1 + (-q0 + q2) * f + (2 * q0 - 5 * q1 + 4 * q2 - q3) * f2 + (-q0 + 3 * q1 - 3 * q2 + q3) * f3);
  const x = cr(p0.x, a.x, b.x, p3.x), y = cr(p0.y, a.y, b.y, p3.y);
  const p = onIce(x, y);
  return [p[0], p[1], b.own, 0];
}

// ---------- game state at a time ----------
const shotCum = [[], []];
SHOTS.forEach((e) => shotCum[e.team].push(e.t));
function countUpTo(arr, T) { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= T) lo = m + 1; else hi = m; } return lo; }
function scoreAt(T) { let s = [0, 0]; for (const g of GOALS) { if (g.t <= T) s = [g.hs, g.as_]; else break; } if (SO && T >= SO.t) s = D.final.slice(); return s; }
const penWin = PENS.map((p) => {
  let end = p.t + 120;
  const g = GOALS.find((g) => g.team !== p.team && g.t > p.t && g.t < end && g.str === "PP");
  if (g) end = g.t;
  return { team: p.team, t0: p.t, t1: end, name: p.name };
});
function shortAt(T) { const c = [0, 0], left = [0, 0]; for (const w of penWin) if (T >= w.t0 && T < w.t1 && T < 3600) { c[w.team]++; left[w.team] = Math.max(left[w.team], w.t1 - T); } return [c, left]; }
function wpAt(T) {
  const tm = (T + 6) / 60, xs = D.wp.t, ys = D.wp.home;
  if (!xs.length) return 0.5;
  if (tm <= xs[0]) return ys[0];
  for (let i = 1; i < xs.length; i++) if (xs[i] >= tm) { const f = (tm - xs[i - 1]) / ((xs[i] - xs[i - 1]) || 1); return ys[i - 1] + (ys[i] - ys[i - 1]) * f; }
  return ys[ys.length - 1];
}
function clockText(T) {
  if (SO && T >= SO.t) return ["SO", "Shootout"];
  if (T >= END && END >= 3600) T = END;
  const per = Math.min(Math.floor(T / 1200) + 1, 99);
  let rem;
  if (per >= 4 && OT3) rem = Math.max(0, 300 - (T - 3600)); else rem = Math.max(0, 1200 - (T - (per - 1) * 1200));
  if (T >= 3600 && T === END) rem = Math.max(0, (OT3 ? 300 : 1200) - (T - 3600 - (per - 4) * 1200));
  const mm = Math.floor(rem / 60), ss = Math.floor(rem % 60);
  const name = per <= 3 ? ["1st", "2nd", "3rd"][per - 1] + " period" : (per === 4 ? (OT3 ? "3-on-3 OT" : "Overtime") : (per - 3) + "OT");
  return [String(mm).padStart(2, "0") + ":" + String(ss).padStart(2, "0"), name];
}
function fmtT(t) { if (t >= 3900 && OT3) return "SO"; const c = clockText(Math.max(0, t - 0.01)); return (t >= 3600 ? (t >= 4800 ? "2OT " : "OT ") : ["P1 ", "P2 ", "P3 "][Math.min(2, Math.floor(t / 1200))]) + c[0]; }

// ---------- play-by-play calls ----------
function pickOne(arr, k) { return arr[Math.abs(k * 7 + 3) % arr.length]; }
function callFor(e, i) {
  const who = jn(e.team, e.name) + esc(e.name), tm = AB[e.team], opp = AB[1 - e.team];
  const g = esc((D.goalies && D.goalies[1 - e.team]) || "the goalie"), last = esc(String(e.name || "").split(" ").slice(-1)[0]);
  if (e.kind === "goal") {
    const before = [e.hs - (e.team === 0 ? 1 : 0), e.as_ - (e.team === 1 ? 1 : 0)];
    const after = [e.hs, e.as_];
    let tail = "";
    if (e.t >= 3600) tail = " OVERTIME WINNER! " + tm + " win it!";
    else if (after[0] === after[1]) tail = " And we're all tied up!";
    else if (before[0] === before[1]) tail = " " + tm + " take the lead!";
    else if (Math.abs(after[0] - after[1]) >= 3) tail = " " + tm + " are rolling!";
    if (e.str && e.str.indexOf("PP") === 0) return "🚨 Power-play goal! " + who + " makes " + opp + " pay!" + tail;
    if (e.str && e.str.indexOf("SH") === 0) return "🚨 SHORTHANDED! " + who + " scores while killing a penalty!" + tail;
    return "🚨 " + pickOne([who + " shoots… HE SCORES!", who + " lets it go… GOAL! What a shot!",
      "SCORE! " + who + " finds the back of the net!", who + " goes top shelf! " + tm + " light the lamp!",
      who + " walks in alone… and BURIES IT!", "Off the rush… " + who + "… GOAL!"], i) + tail;
  }
  if (e.kind === "shot") return pickOne([who + " fires… turned aside by " + g + ".", who + " lets it fly — glove save, " + g + "!",
    "Big stop by " + g + " on " + last + "!", who + " from the slot… kicked out by " + g + "!",
    who + " tries his luck — " + g + " says no.", last + " rips one on net — " + g + " is there!"], i);
  if (e.kind === "pen") return pickOne([who + " heads to the box for two minutes. " + opp + " on the power play!",
    "Whistle! " + who + " is called for a minor — " + opp + " go on the man advantage.",
    "Two minutes for " + who + ". Big chance for " + opp + "!"], i);
  if (e.kind === "so") return "🥅 " + tm + " win it in the shootout!";
  return "";
}
let lastCall = -2, callAt = 0;
function updateCall(T) {
  let idx = -1;
  for (let i = 0; i < EV.length && EV[i].t <= T; i++) idx = i;
  const done = T >= END && END > 0 && !playing && T > 0;
  const key = done ? -99 : idx;
  const el = $("call");
  if (key !== lastCall) {
    lastCall = key; callAt = performance.now();
    if (done) {
      const s = D.final, w = s[0] > s[1] ? 0 : 1;
      el.innerHTML = "🎙️ That's the final buzzer! " + AB[w] + " win " + Math.max(s[0], s[1]) + "–" + Math.min(s[0], s[1]) + (D.so ? " in a shootout" : D.ot ? " in overtime" : "") + ".";
      el.className = "goal";
    } else if (idx < 0) { el.innerHTML = T > 0 ? "🎙️ Puck drop! Here we go!" : "🎙️ Welcome to the rink! Press ▶ for puck drop."; el.className = ""; }
    else { const e = EV[idx]; el.innerHTML = "🎙️ " + callFor(e, idx); el.className = e.kind === "goal" || e.kind === "so" ? "goal" : (e.kind === "pen" ? "pen" : ""); }
  } else if (!done && performance.now() - callAt > 4500) el.classList.add("dim");
}
function dotNum(T, tm, k, own) {
  const L = LINES[tm];
  if (!L || !L.F.length) return "";
  if (k === 0 && own === tm) {
    const nx = SHOTS.find((e) => e.t >= T);
    if (nx && nx.team === tm && nx.t - T < 2.4 && NUM[tm][nx.name]) return NUM[tm][nx.name];
  }
  const sh = Math.floor(T / 45), fl = (sh % 4) * 3, dp = (sh % 3) * 2;
  const F = L.F.slice(fl, fl + 3), Dd = L.D.slice(dp, dp + 2);
  const order = teamCount(T, tm) <= 3 ? [F[0], F[1], Dd[0]] : [F[0], F[1], F[2], Dd[0], Dd[1]];
  return order[k] || "";
}

// ---------- three stars ----------
function stars() {
  const pts = {};
  const add = (name, team, g, a) => { if (!name) return; const k = team + "|" + name; pts[k] = pts[k] || { name, team, g: 0, a: 0, s: 0 }; pts[k].g += g; pts[k].a += a; pts[k].s += 2.2 * g + a; };
  GOALS.forEach((g) => { add(g.name, g.team, 1, 0); (g.ast || []).forEach((a) => add(a, g.team, 0, 1)); });
  const fin = D.final, winner = fin[0] > fin[1] ? 0 : 1;
  [0, 1].forEach((tm) => {
    const ga = fin[1 - tm] - (SO && winner !== tm ? 0 : 0);
    const allowed = tm === 0 ? fin[1] : fin[0];
    if (allowed <= 1 && D.goalies && D.goalies[tm]) pts["g" + tm] = { name: D.goalies[tm], team: tm, goalie: true, ga: allowed, s: allowed === 0 ? 4.5 : 2.5 + (winner === tm ? 0.5 : 0) };
  });
  return Object.values(pts).sort((a, b) => b.s - a.s || b.g - a.g).slice(0, 3);
}

// ---------- layout ----------
const app = $("app"), rink = $("rink"), tl = $("tl");
const ctx = rink.getContext("2d"), tctx = tl.getContext("2d");
let W = 600, H = 260, DPR = 1, viewW = 204, scale = 3, camX = 0, follow = null;
let off = null;            // cached rink drawing
const logoImg = new Image(); let logoOK = false;
logoImg.onload = () => { logoOK = true; drawRinkCache(); };
if (D.home_logo_ice) logoImg.src = D.home_logo_ice;
$("hlogo").src = D.home_logo || ""; $("alogo").src = D.away_logo || "";
$("hlogo").onerror = function () { this.style.display = "none"; };
$("alogo").onerror = function () { this.style.display = "none"; };
$("hab").textContent = AB[0]; $("aab").textContent = AB[1];
$("hch").style.background = COL[0]; $("ach").style.background = COL[1];

function layout() {
  W = Math.max(280, app.clientWidth);
  DPR = Math.min(window.devicePixelRatio || 1, 2.5);
  if (follow === null) follow = W < 640;
  viewW = follow ? Math.min(204, Math.max(118, W / 3.1)) : 204;
  scale = W / viewW;
  H = Math.round(90 * scale);
  rink.width = Math.round(W * DPR); rink.height = Math.round(H * DPR); rink.style.height = H + "px";
  tl.width = Math.round(W * DPR); tl.height = Math.round(64 * DPR);
  $("b_cam").textContent = follow ? "🎥 Follow" : "🎥 Full rink";
  drawRinkCache();
}
function rr(c, x, y, w, h, r) { c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath(); }
function drawRinkCache() {
  const ps = scale * DPR;
  off = document.createElement("canvas");
  off.width = Math.round(204 * ps); off.height = Math.round(90 * ps);
  const c = off.getContext("2d");
  const X = (x) => (x + 102) * ps, Y = (y) => (45 - y) * ps;
  c.fillStyle = "#0b1220"; c.fillRect(0, 0, off.width, off.height);
  const g = c.createLinearGradient(0, Y(42.5), 0, Y(-42.5));
  g.addColorStop(0, "#f4f8ff"); g.addColorStop(0.5, "#ffffff"); g.addColorStop(1, "#eaf1fb");
  rr(c, X(-100), Y(42.5), 200 * ps, 85 * ps, 28 * ps); c.fillStyle = g; c.fill();
  c.save(); rr(c, X(-100), Y(42.5), 200 * ps, 85 * ps, 28 * ps); c.clip();
  c.globalAlpha = 0.05; c.fillStyle = "#7aa7d9";
  for (let i = 0; i < 24; i++) { c.beginPath(); c.ellipse(X(-90 + i * 8), Y(Math.sin(i * 1.7) * 30), 14 * ps, 3 * ps, i, 0, 6.283); c.fill(); }
  c.globalAlpha = 1;
  const line = (x, col, w) => { c.fillStyle = col; c.fillRect(X(x) - w * ps / 2, Y(42.5), w * ps, 85 * ps); };
  line(0, "#c8102e", 1.0); line(-25, "#1d4fa3", 1.0); line(25, "#1d4fa3", 1.0);
  line(-89, "#c8102e", 0.17); line(89, "#c8102e", 0.17);
  c.lineWidth = Math.max(1, 0.17 * ps);
  const circ = (x, y, r, col, fill) => { c.beginPath(); c.arc(X(x), Y(y), r * ps, 0, 6.283); if (fill) { c.fillStyle = col; c.fill(); } else { c.strokeStyle = col; c.stroke(); } };
  circ(0, 0, 15, "#1d4fa3"); circ(0, 0, 0.5, "#1d4fa3", true);
  if (logoOK) {
    c.save(); c.globalAlpha = 0.9; const s = 21 * ps, ar = (logoImg.naturalWidth || 1) / (logoImg.naturalHeight || 1);
    const w = ar >= 1 ? s : s * ar, h = ar >= 1 ? s / ar : s;
    try { c.drawImage(logoImg, X(0) - w / 2, Y(0) - h / 2, w, h); } catch (e) {}
    c.restore();
  } else if (D.home) {
    c.fillStyle = COL[0] + "33"; c.font = "900 " + (8 * ps) + "px Arial"; c.textAlign = "center"; c.textBaseline = "middle"; c.fillText(D.home, X(0), Y(0));
  }
  for (const x of [-69, 69]) for (const y of [-22, 22]) {
    circ(x, y, 15, "#c8102e"); circ(x, y, 1, "#c8102e", true);
    c.strokeStyle = "#c8102e";
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) { c.beginPath(); c.moveTo(X(x + sx * 2), Y(y + sy * 15)); c.lineTo(X(x + sx * 2), Y(y + sy * 17)); c.stroke(); }
  }
  for (const x of [-20, 20]) for (const y of [-22, 22]) circ(x, y, 1, "#c8102e", true);
  for (const sgn of [-1, 1]) {
    c.beginPath();
    if (sgn > 0) c.arc(X(89), Y(0), 6 * ps, Math.PI / 2, Math.PI * 1.5); else c.arc(X(-89), Y(0), 6 * ps, -Math.PI / 2, Math.PI / 2);
    c.closePath(); c.fillStyle = "#6fb2ff99"; c.fill(); c.strokeStyle = "#c8102e"; c.stroke();
    c.fillStyle = "#f2f2f2"; c.strokeStyle = "#c8102e"; c.lineWidth = Math.max(1.2, 0.3 * ps);
    rr(c, Math.min(X(sgn * 89), X(sgn * 92.4)), Y(3), Math.abs(X(sgn * 92.4) - X(sgn * 89)), 6 * ps, 1.2 * ps); c.fill(); c.stroke();
    c.lineWidth = Math.max(1, 0.17 * ps);
  }
  c.restore();
  rr(c, X(-100), Y(42.5), 200 * ps, 85 * ps, 28 * ps); c.lineWidth = 1.1 * ps; c.strokeStyle = "#c9d3e6"; c.stroke();
}

// ---------- skaters ----------
const players = [[], []];
function teamCount(T, tm) {
  let n = (T >= 3600 && OT3 && !(SO && T >= SO.t)) ? 3 : 5;
  const [c] = shortAt(T); n -= Math.min(2, c[tm]);
  return n;
}
function targets(T, px, py, own, tm, n) {
  const d = DIR(tm), u = d * px, sy = py >= 0 ? 1 : -1, out = [];
  const W2 = (uu, yy) => onIce(d * uu, yy);
  if (own === tm) {                                 // attacking
    out.push(W2(u - 2.2, py + 1.2));
    out.push(W2(Math.min(u + 9, 84), clamp(-py * 0.4 + sy * -12, -32, 32)));
    out.push(W2(Math.min(Math.max(u - 2, -20) + 6, 78), clamp(py * 0.2 + sy * 16, -34, 34)));
    out.push(W2(clamp(u - 32, -60, 38), 17));
    out.push(W2(clamp(u - 32, -60, 38), -17));
  } else if (own === 1 - tm) {                      // defending (own net at u = -89)
    const mid = u + (-89 - u) * 0.42;
    out.push(W2(u - 2.5, py - sy * 2.5));
    out.push(W2(clamp(mid, -84, 60), clamp(py * 0.5 + 10, -30, 30)));
    out.push(W2(clamp(mid, -84, 60), clamp(py * 0.5 - 10, -30, 30)));
    out.push(W2(clamp(u - 14, -70, 50), clamp(-sy * 20, -32, 32)));
    out.push(W2(clamp(u - 18, -70, 50), clamp(sy * 24, -32, 32)));
  } else {                                          // loose puck
    out.push(W2(u - 4, py + 3)); out.push(W2(u - 12, py - 12)); out.push(W2(u - 14, py + 14));
    out.push(W2(clamp(u - 30, -65, 30), 16)); out.push(W2(clamp(u - 30, -65, 30), -16));
  }
  if (n <= 3) return [out[0], out[1], out[3]];
  return out.slice(0, n);
}
function updatePlayers(T, dtGame, snap, cel) {
  const [px, py, own] = puckAt(T);
  for (const tm of [0, 1]) {
    const n = teamCount(T, tm);
    let tg = targets(T, px, py, own, tm, n);
    if (cel && cel.team === tm) tg = tg.map((p, k) => onIce(cel.x - DIR(tm) * (3 + k * 1.2), cel.y + Math.cos(k * 2.1) * 3.2));
    const ps = players[tm];
    while (ps.length < tg.length) ps.push({ x: tg[ps.length][0], y: tg[ps.length][1], a: 0 });
    for (let k = 0; k < ps.length; k++) {
      const p = ps[k];
      if (k < tg.length) {
        const a = snap ? 1 : 1 - Math.exp(-dtGame * (k === 0 ? 3.2 : 1.6));
        p.x += (tg[k][0] - p.x) * a; p.y += (tg[k][1] - p.y) * a; p.a = Math.min(1, p.a + (snap ? 1 : 0.12));
      } else { p.a = Math.max(0, p.a - (snap ? 1 : 0.12)); }
    }
  }
}

// ---------- effects ----------
let confetti = [], flashes = [];
function boom(team, x, y) {
  confetti = [];
  const cols = [COL[team], "#ffffff", "#ffd84d"];
  for (let i = 0; i < 140; i++) confetti.push({ x, y, vx: (Math.random() - 0.5) * 70, vy: Math.random() * 55 + 10, r: Math.random() * 6.283, vr: (Math.random() - 0.5) * 10, c: cols[i % 3], life: 1 });
}
let audio = null, sound = false;
function horn() {
  if (!sound) return;
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)();
    const t0 = audio.currentTime, g = audio.createGain(), lp = audio.createBiquadFilter();
    lp.type = "lowpass"; lp.frequency.value = 900; g.connect(audio.destination); lp.connect(g);
    g.gain.setValueAtTime(0.0001, t0); g.gain.exponentialRampToValueAtTime(0.35, t0 + 0.08);
    g.gain.setValueAtTime(0.35, t0 + 1.4); g.gain.exponentialRampToValueAtTime(0.0001, t0 + 2.0);
    for (const f of [116.5, 174.6, 233]) { const o = audio.createOscillator(); o.type = "sawtooth"; o.frequency.value = f; o.connect(lp); o.start(t0); o.stop(t0 + 2.05); }
  } catch (e) {}
}

// ---------- drawing ----------
const trail = [];
function W2S(x, y) { return [(x - camX + viewW / 2) * scale, (45 - y) * scale]; }
function draw(T, now, cel) {
  const [px, py, own, moving] = puckAt(T);
  if (follow) { const target = clamp(px, -102 + viewW / 2, 102 - viewW / 2); camX += (target - camX) * 0.12; if (Math.abs(target - camX) > 60) camX = target; }
  else camX = 0;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (off) ctx.drawImage(off, (camX - viewW / 2 + 102) * scale * DPR, 0, viewW * scale * DPR, 90 * scale * DPR, 0, 0, rink.width, rink.height);
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  // recent shots: fading lines to the net
  for (const e of SHOTS) {
    if (e.t > T) break;
    const age = T - e.t;
    if (age > 2.2) continue;
    const d = DIR(e.team), a = 1 - age / 2.2;
    const [x1, y1] = W2S(e.x, e.y), [x2, y2] = W2S(d * 89, 0);
    ctx.strokeStyle = (e.kind === "goal" ? "#ff2d2d" : COL[e.team]); ctx.globalAlpha = 0.55 * a; ctx.lineWidth = e.kind === "goal" ? 3 : 2;
    ctx.setLineDash([6, 5]); ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.setLineDash([]);
    ctx.globalAlpha = a; ctx.fillStyle = e.kind === "goal" ? "#ff2d2d" : COL[e.team];
    ctx.beginPath(); ctx.arc(x1, y1, 3.2, 0, 6.283); ctx.fill();
    if (e.kind === "shot" && age < 1.4) {
      const [nx, ny] = W2S(d * 80, 9);
      ctx.font = "800 " + Math.max(10, 3.4 * scale) + "px Arial"; ctx.textAlign = "center"; ctx.fillStyle = COL[1 - e.team];
      ctx.fillText("SAVE", nx, ny);
    }
    ctx.globalAlpha = 1;
  }
  // goal light
  const lit = cel || (GOALS.find((g) => T >= g.t && T - g.t < 1.5));
  if (lit) {
    const d = DIR(lit.team), [gx, gy] = W2S(d * 93.5, 0), pulse = 0.55 + 0.45 * Math.sin(now / 90);
    const rg = ctx.createRadialGradient(gx, gy, 2, gx, gy, 18 * scale);
    rg.addColorStop(0, "rgba(255,40,40," + (0.75 * pulse) + ")"); rg.addColorStop(1, "rgba(255,40,40,0)");
    ctx.fillStyle = rg; ctx.beginPath(); ctx.arc(gx, gy, 18 * scale, 0, 6.283); ctx.fill();
  }
  // goalies
  for (const tm of [0, 1]) {
    const d = DIR(tm), gxw = -d * 86.2, gyw = clamp(py * 0.12, -2.6, 2.6);
    const [gx, gy] = W2S(gxw, gyw);
    const r = Math.max(4.5, 2.5 * scale);
    ctx.fillStyle = "#0008"; ctx.beginPath(); ctx.ellipse(gx + 1.5, gy + 2, r, r * 0.8, 0, 0, 6.283); ctx.fill();
    ctx.fillStyle = COL[tm]; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.rect(gx - r * 0.75, gy - r, r * 1.5, r * 2); ctx.fill(); ctx.stroke();
    if (GNUM[tm] && r >= 5.5) { ctx.font = "800 " + Math.round(r * 0.95) + "px Arial"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.lineWidth = 2.4; ctx.strokeStyle = "#000b"; ctx.strokeText(GNUM[tm], gx, gy + 0.5); ctx.fillStyle = "#fff"; ctx.fillText(GNUM[tm], gx, gy + 0.5); ctx.textBaseline = "alphabetic"; }
  }
  // skaters
  for (const tm of [0, 1]) for (let k = 0; k < players[tm].length; k++) {
    const p = players[tm][k];
    if (p.a <= 0.01) continue;
    const [sx, sy] = W2S(p.x, p.y), r = Math.max(4.5, 2.6 * scale);
    ctx.globalAlpha = p.a;
    ctx.fillStyle = "#00000040"; ctx.beginPath(); ctx.ellipse(sx + 1.5, sy + 2.2, r, r * 0.75, 0, 0, 6.283); ctx.fill();
    ctx.fillStyle = COL[tm]; ctx.strokeStyle = "#ffffff"; ctx.lineWidth = Math.max(1.5, 0.45 * scale);
    ctx.beginPath(); ctx.arc(sx, sy, r, 0, 6.283); ctx.fill(); ctx.stroke();
    if (k === 0 && own === tm) { ctx.strokeStyle = "#ffd84d"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(sx, sy, r + 3, 0, 6.283); ctx.stroke(); }
    const lab = r >= 5.5 ? dotNum(T, tm, k, own) : "";
    if (lab) { ctx.font = "800 " + Math.round(r * (lab.length > 1 ? 0.95 : 1.1)) + "px Arial"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.lineWidth = 2.4; ctx.strokeStyle = "#000b"; ctx.strokeText(lab, sx, sy + 0.5); ctx.fillStyle = "#fff"; ctx.fillText(lab, sx, sy + 0.5); ctx.textBaseline = "alphabetic"; }
    ctx.globalAlpha = 1;
  }
  // puck + trail
  const [qx, qy] = W2S(px, py);
  trail.push([px, py]); if (trail.length > 6) trail.shift();
  for (let i = 0; i < trail.length - 1; i++) {
    const [ax, ay] = W2S(trail[i][0], trail[i][1]), [bx, by] = W2S(trail[i + 1][0], trail[i + 1][1]);
    if (Math.hypot(bx - ax, by - ay) > 45) continue;
    ctx.strokeStyle = "rgba(20,20,30," + (0.05 + i * 0.035) + ")"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  }
  const pr = Math.max(3, 1.15 * scale);
  ctx.fillStyle = "#111"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.ellipse(qx, qy, pr, pr * 0.8, 0, 0, 6.283); ctx.fill(); ctx.stroke();
  // shooter name tag just before a shot
  const nxt = SHOTS.find((e) => e.t >= T);
  if (nxt && nxt.t - T < 2.4 && own === nxt.team && nxt.name) {
    const label = nxt.name.split(" ").slice(-1)[0];
    ctx.font = "700 " + (W < 600 ? 11 : 12) + "px -apple-system, Segoe UI, Arial"; ctx.textAlign = "center";
    const tw = ctx.measureText(label).width + 10;
    ctx.fillStyle = "#0b1220d9"; rr(ctx, qx - tw / 2, qy - 26, tw, 16, 6); ctx.fill();
    ctx.fillStyle = "#fff"; ctx.textBaseline = "middle"; ctx.fillText(label, qx, qy - 18); ctx.textBaseline = "alphabetic";
  }
  // confetti
  if (confetti.length) {
    for (const c of confetti) {
      const [cx, cy] = W2S(c.x, c.y);
      ctx.save(); ctx.globalAlpha = Math.max(0, c.life); ctx.translate(cx, cy); ctx.rotate(c.r); ctx.fillStyle = c.c; ctx.fillRect(-3, -1.5, 6, 3); ctx.restore();
    }
  }
}
function stepConfetti(dt) {
  for (const c of confetti) { c.vy -= 60 * dt; c.x += c.vx * dt; c.y += c.vy * dt * 0.6; c.r += c.vr * dt; c.life -= dt * 0.45; }
  confetti = confetti.filter((c) => c.life > 0);
}

function drawTimeline(T) {
  const w = tl.width / DPR, h = 64, pad = 8;
  tctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  tctx.clearRect(0, 0, w, h);
  const X = (t) => pad + (t / END) * (w - 2 * pad), mid = h / 2, amp = h / 2 - 7;
  tctx.fillStyle = "#13203a"; rr(tctx, 0, 0, w, h, 8); tctx.fill();
  const xs = D.wp.t, ys = D.wp.home;
  const pts = xs.map((m, i) => [X(Math.min(m * 60, END)), mid - (ys[i] - 0.5) * 2 * amp]);
  for (const [tm, clipTop] of [[0, true], [1, false]]) {
    tctx.save(); tctx.beginPath(); if (clipTop) tctx.rect(0, 0, w, mid); else tctx.rect(0, mid, w, h); tctx.clip();
    tctx.beginPath(); tctx.moveTo(X(0), mid); pts.forEach((p) => tctx.lineTo(p[0], p[1])); tctx.lineTo(pts.length ? pts[pts.length - 1][0] : X(END), mid); tctx.closePath();
    tctx.fillStyle = COL[tm]; tctx.globalAlpha = 0.85; tctx.fill(); tctx.restore();
  }
  tctx.globalAlpha = 1; tctx.strokeStyle = "#e8eefcb0"; tctx.lineWidth = 1.4;
  tctx.beginPath(); pts.forEach((p, i) => (i ? tctx.lineTo(p[0], p[1]) : tctx.moveTo(p[0], p[1]))); tctx.stroke();
  tctx.globalAlpha = 1; tctx.strokeStyle = "#ffffff30"; tctx.lineWidth = 1;
  tctx.beginPath(); tctx.moveTo(X(0), mid); tctx.lineTo(X(END), mid); tctx.stroke();
  tctx.font = "10px Arial"; tctx.fillStyle = "#8ea0c4"; tctx.textAlign = "left";
  const marks = [[0, "1st"], [1200, "2nd"], [2400, "3rd"]]; if (END > 3600) marks.push([3600, "OT"]);
  for (const [t, lb] of marks) { if (t >= 3600 && X(END) - X(t) < 18) continue; if (t > 0) { tctx.strokeStyle = "#ffffff26"; tctx.beginPath(); tctx.moveTo(X(t), 2); tctx.lineTo(X(t), h - 2); tctx.stroke(); } tctx.fillText(lb, X(t) + 3, h - 4); }
  for (const p of PENS) { tctx.fillStyle = "#f5c518"; const x = X(p.t); tctx.beginPath(); tctx.moveTo(x, h - 2); tctx.lineTo(x - 3.5, h - 8); tctx.lineTo(x + 3.5, h - 8); tctx.fill(); }
  for (const g of GOALS) { const x = X(g.t), y = g.team === 0 ? 9 : h - 15; tctx.fillStyle = COL[g.team]; tctx.strokeStyle = "#fff"; tctx.lineWidth = 1.5; tctx.beginPath(); tctx.arc(x, y, 5, 0, 6.283); tctx.fill(); tctx.stroke(); }
  const x = X(Math.min(T, END));
  tctx.fillStyle = "#ffffff12"; tctx.fillRect(X(0), 0, x - X(0), h);
  tctx.strokeStyle = "#ffd84d"; tctx.lineWidth = 2; tctx.beginPath(); tctx.moveTo(x, 0); tctx.lineTo(x, h); tctx.stroke();
  tctx.fillStyle = "#ffd84d"; tctx.beginPath(); tctx.arc(x, mid - (wpAt(T) - 0.5) * 2 * amp, 4.5, 0, 6.283); tctx.fill();
}

// ---------- HUD ----------
let lastScore = "0-0";
function hud(T) {
  const s = scoreAt(T);
  $("hsc").textContent = s[0]; $("asc").textContent = s[1];
  const key = s.join("-");
  if (key !== lastScore) { const which = s[0] !== +lastScore.split("-")[0] ? "hsc" : "asc"; $(which).classList.add("pop"); setTimeout(() => $(which).classList.remove("pop"), 700); lastScore = key; }
  $("hsog").textContent = "SOG " + countUpTo(shotCum[0], T); $("asog").textContent = "SOG " + countUpTo(shotCum[1], T);
  const [c, p] = clockText(T); $("clk").textContent = c; $("per").textContent = T >= END && !playing && END > 0 && T > 0 ? "Final" + (D.so ? " (SO)" : D.ot ? " (OT)" : "") : p;
  const [cnt, left] = shortAt(T); const pp = $("pp");
  if (cnt[0] !== cnt[1]) { const tm = cnt[0] > cnt[1] ? 1 : 0; const l = Math.ceil(left[1 - tm]); pp.style.visibility = "visible"; pp.style.background = COL[tm]; pp.style.color = "#fff";
    pp.textContent = "PP " + AB[tm] + " " + Math.floor(l / 60) + ":" + String(l % 60).padStart(2, "0") + ((cnt[0] + cnt[1] > 1 && Math.abs(cnt[0] - cnt[1]) > 1) ? " 5-on-3" : ""); }
  else pp.style.visibility = "hidden";
  const w = wpAt(T); $("wph").textContent = AB[0] + " " + Math.round(100 * w) + "%"; $("wpa").textContent = AB[1] + " " + Math.round(100 * (1 - w)) + "%";
  // ticker
  const recent = [];
  for (let i = EV.length - 1; i >= 0 && recent.length < 4; i--) if (EV[i].t <= T) recent.push(EV[i]);
  $("tick").innerHTML = recent.map((e) => {
    const team = AB[e.team];
    let txt, cls = "";
    if (e.kind === "goal") { cls = "g"; txt = "🚨 " + team + " GOAL — " + jn(e.team, e.name) + esc(e.name) + (e.ast && e.ast.length ? " (" + e.ast.map(esc).join(", ") + ")" : " (unassisted)") + (e.str && e.str !== "EV" ? " · " + e.str : "") + " · " + AB[1] + " " + e.as_ + "–" + e.hs + " " + AB[0]; }
    else if (e.kind === "pen") { cls = "p"; txt = "⛔ " + team + " penalty — " + jn(e.team, e.name) + esc(e.name) + " (2:00)"; }
    else if (e.kind === "so") { cls = "g"; txt = "🥅 " + team + " win the shootout"; }
    else txt = team + " shot — " + jn(e.team, e.name) + esc(e.name) + " · saved";
    return '<div class="' + cls + '"><span class="tt">' + fmtT(e.t) + "</span>" + txt + "</div>";
  }).join("") || '<div>Puck drop! Press ▶ to start.</div>';
  updateCall(T);
}
function esc(s) { return String(s || "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

// ---------- playback ----------
const SPEEDS = [1, 2, 5, 10, 20];
let speed = 5, playing = false, T = 0, freezeUntil = 0, cel = null, card = null, lastNow = performance.now();
SPEEDS.forEach((s) => { const b = document.createElement("button"); b.className = "btn" + (s === speed ? " on" : ""); b.textContent = s + "×"; b.onclick = () => setSpeed(s); $("speeds").appendChild(b); });
function setSpeed(s) { speed = s; [...$("speeds").children].forEach((b) => b.classList.toggle("on", b.textContent === s + "×")); }
function setPlaying(p) {
  playing = p; $("b_play").textContent = p ? "⏸ Pause" : "▶ Play"; $("big").style.display = p ? "none" : "flex";
  $("bigbtn").textContent = T >= END ? "↻" : "▶";
  if (p && T >= END) seek(0);
}
function showBanner(team, t1, t2, t3) {
  const b = $("banner"); b.style.background = "linear-gradient(135deg," + COL[team] + ", #0b1220)";
  b.querySelector(".t1").textContent = t1; b.querySelector(".t2").textContent = t2; b.querySelector(".t3").textContent = t3 || "";
  b.classList.add("show");
}
function hideBanner() { $("banner").classList.remove("show"); }
function showCard(html, ms) { $("cardbox").innerHTML = html; $("card").classList.add("show"); card = ms ? performance.now() + ms : Infinity; }
function hideCard() { $("card").classList.remove("show"); card = null; }
function seek(t) {
  T = clamp(t, 0, END); cel = null; freezeUntil = 0; confetti = []; hideBanner(); hideCard(); trail.length = 0;
  updatePlayers(T, 0, true, null);
  const [px] = puckAt(T); camX = follow ? clamp(px, -102 + viewW / 2, 102 - viewW / 2) : 0;
  lastScore = scoreAt(T).join("-");
  render(performance.now());
}
function finalCard() {
  const s = D.final, st = stars();
  const win = s[0] > s[1] ? 0 : 1;
  const starRows = st.map((p, i) => '<div class="row">' + "★".repeat(3 - i) + "&nbsp; <b>" + (p.goalie ? (GNUM[p.team] ? "#" + GNUM[p.team] + " " : "") : jn(p.team, p.name)) + esc(p.name) + "</b> (" + AB[p.team] + ") " +
    (p.goalie ? (p.ga === 0 ? "shutout" : p.ga + " GA") : (p.g ? p.g + "G " : "") + (p.a ? p.a + "A" : "")) + "</div>").join("");
  showCard("<h3>Final" + (D.so ? " / SO" : D.ot ? " / OT" : "") + "</h3>" +
    '<div class="row" style="font-size:22px;font-weight:900;color:#fff">' + AB[1] + " " + s[1] + " – " + s[0] + " " + AB[0] + "</div>" +
    '<div class="row">Shots ' + AB[1] + " " + shotCum[1].length + " – " + shotCum[0].length + " " + AB[0] + "</div>" +
    (starRows ? '<div class="stars"><div class="row" style="color:#8ea0c4">Three stars</div>' + starRows + "</div>" : "") +
    '<button class="btn" id="replay">↻ Watch again</button>', 0);
  const rb = document.getElementById("replay"); if (rb) rb.onclick = () => { seek(0); setPlaying(true); };
}
function render(now) {
  draw(T, now, cel); drawTimeline(T); hud(T);
}
function frame(now) {
  const dtReal = Math.min(0.1, (now - lastNow) / 1000); lastNow = now;
  let dtGame = 0;
  if (card && now > card) hideCard();
  if (playing) {
    if (now < freezeUntil) { dtGame = dtReal * 1.5; }
    else {
      if (cel) { cel = null; hideBanner(); }
      if (card === Infinity) hideCard();
      const prev = T; T = Math.min(END, T + dtReal * speed * 5); dtGame = T - prev;
      const g = GOALS.find((g) => g.t > prev && g.t <= T);
      const brk = [1200, 2400, 3600].find((b) => b > prev && b <= T && b < END);
      if (g) {
        T = g.t; const d = DIR(g.team); cel = { team: g.team, x: d * 80, y: clamp(g.y, -20, 20) };
        freezeUntil = now + (speed >= 10 ? 1700 : 2600); boom(g.team, d * 85, 0); horn();
        showBanner(g.team, g.str === "EN" ? "EMPTY NET!" : (g.t >= 3600 ? "OT WINNER!" : "GOAL!"), AB[g.team] + " · " + g.name,
          (g.ast && g.ast.length ? "Assists: " + g.ast.join(", ") : "Unassisted") + (g.str && !["EV", "3v3", "EN"].includes(g.str) ? " · " + g.str + " goal" : ""));
      } else if (brk) {
        T = brk; freezeUntil = now + 1700; const s = scoreAt(T);
        showCard("<h3>End of " + (brk === 1200 ? "1st" : brk === 2400 ? "2nd" : "regulation") + "</h3>" +
          '<div class="row" style="font-size:20px;font-weight:900;color:#fff">' + AB[1] + " " + s[1] + " – " + s[0] + " " + AB[0] + "</div>" +
          '<div class="row">Shots ' + AB[1] + " " + countUpTo(shotCum[1], T) + " – " + countUpTo(shotCum[0], T) + " " + AB[0] + "</div>" +
          (brk === 3600 ? '<div class="row">' + (OT3 ? "Heading to 3-on-3 overtime" : "Sudden-death overtime") + "</div>" : ""), 1700);
      } else if (SO && prev < SO.t && T >= SO.t) {
        freezeUntil = now + 2200; boom(SO.team, DIR(SO.team) * 85, 0); horn();
        showBanner(SO.team, "SHOOTOUT!", AB[SO.team] + " win the shootout", "");
      }
      if (T >= END && now >= freezeUntil) { setPlaying(false); finalCard(); }
    }
  }
  if (playing || confetti.length || now < freezeUntil + 500 || cel) {
    updatePlayers(T, Math.max(dtGame, cel ? dtReal * 1.2 : 0), false, cel);
    stepConfetti(dtReal);
    render(now);
  }
  requestAnimationFrame(frame);
}

// ---------- controls ----------
$("b_play").onclick = () => setPlaying(!playing);
$("bigbtn").onclick = () => setPlaying(true);
rink.onclick = () => setPlaying(!playing);
$("b_restart").onclick = () => { seek(0); };
$("b_next").onclick = () => { const g = GOALS.find((g) => g.t - 8 > T + 0.5); seek(g ? g.t - 8 : END); if (!playing && g) setPlaying(true); };
$("b_prev").onclick = () => { const gs = GOALS.filter((g) => g.t - 8 < T - 3); const g = gs[gs.length - 1]; seek(g ? g.t - 8 : 0); };
$("b_cam").onclick = () => { follow = !follow; layout(); seek(T); };
$("b_snd").onclick = () => { sound = !sound; $("b_snd").textContent = sound ? "🔊" : "🔇"; if (sound) { audio = audio || new (window.AudioContext || window.webkitAudioContext)(); audio.resume && audio.resume(); } };
if (!document.fullscreenEnabled) $("b_fs").style.display = "none";
$("b_fs").onclick = () => { if (document.fullscreenElement) document.exitFullscreen(); else app.requestFullscreen && app.requestFullscreen(); };
document.addEventListener("fullscreenchange", () => setTimeout(() => { layout(); seek(T); }, 60));
let dragging = false;
function tlSeek(ev) { const r = tl.getBoundingClientRect(); const f = clamp((ev.clientX - r.left - 8) / (r.width - 16), 0, 1); seek(f * END); }
tl.addEventListener("pointerdown", (e) => { dragging = true; tl.setPointerCapture(e.pointerId); tlSeek(e); });
tl.addEventListener("pointermove", (e) => { if (dragging) tlSeek(e); });
tl.addEventListener("pointerup", () => { dragging = false; });
app.addEventListener("keydown", (e) => {
  if (e.key === " ") { e.preventDefault(); setPlaying(!playing); }
  else if (e.key === "ArrowRight") $("b_next").click();
  else if (e.key === "ArrowLeft") $("b_prev").click();
  else if (e.key === "+" || e.key === "=") setSpeed(SPEEDS[Math.min(SPEEDS.length - 1, SPEEDS.indexOf(speed) + 1)]);
  else if (e.key === "-") setSpeed(SPEEDS[Math.max(0, SPEEDS.indexOf(speed) - 1)]);
});
let rzT = null;
new ResizeObserver(() => { clearTimeout(rzT); rzT = setTimeout(() => { const f = follow; layout(); follow = f; seek(T); }, 80); }).observe(app);
layout(); seek(0);
requestAnimationFrame(frame);
window.__anim = { seek, setPlaying, get T() { return T; }, K, END };
})();
</script>
"""


def animation_html(res):
    """Self-contained HTML/JS game player (canvas): skaters, puck, goal celebrations, win-probability scrubber."""
    f = res["featured"]
    home, away = res["home"], res["away"]
    ch, ca = team_colors(home, away)
    events = []
    for (_sim, t, team, kind, pl, a1, a2, x, y, hs, as_, strength) in f["events"]:
        nm = f["names"][team]
        rx, ry = (x, y) if team == 0 else (-x, -y)
        events.append(dict(
            t=int(t), team=int(team), kind=kind,
            name=nm[pl] if 0 <= pl < len(nm) else "",
            ast=[nm[a] for a in (a1, a2) if 0 <= a < len(nm)],
            x=round(float(np.clip(rx, -95, 95)), 1), y=round(float(np.clip(ry, -38, 38)), 1),
            hs=int(hs), as_=int(as_), str=str(strength)))
    end = max([e["t"] for e in events if e["kind"] != "so"] + [3600])
    if any(e["kind"] == "so" for e in events):
        end = 3900
    data = dict(home=home, away=away, hc=ch, ac=ca, events=events, end=end,
                final=[int(v) for v in f["final"]], ot=bool(f["ot"]), so=bool(f["so"]),
                playoff=bool(res.get("playoff")),
                home_logo=logo_url(home, dark=True), away_logo=logo_url(away, dark=True),
                home_logo_ice=logo_url(home), goalies=[res["goalies"][0]["Goalie"], res["goalies"][1]["Goalie"]],
                wp=dict(t=[float(v) for v in res["winprob"]["t"]], home=[float(v) for v in res["winprob"]["home"]]))
    jn = jersey_numbers()
    nums, lines_ = [], []
    for k in (0, 1):
        b = res["props"][k]
        ids = [str(i) for i in b["ids"]]
        nums.append({b["names"][i]: jn[ids[i]] for i in range(len(ids)) if jn.get(ids[i])})
        lines_.append(dict(F=[jn.get(ids[i], "") for i in range(len(ids)) if b["pos"][i] != "D"][:12],
                           D=[jn.get(ids[i], "") for i in range(len(ids)) if b["pos"][i] == "D"][:6]))
    data.update(nums=nums, lines=lines_, gnums=[jn.get(str(res["props"][k].get("goalie_id")), "") for k in (0, 1)])
    return ANIM_TEMPLATE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))


def flow_figure(res):
    """Real-style game flow chart: cumulative shots, goals, and power plays."""
    f = res["featured"]
    home, away = res["home"], res["away"]
    ch, ca = team_colors(home, away)
    end = max([e[1] for e in f["events"]] + [3600])
    fig = go.Figure()
    for team, col, ab in ((0, ch, home), (1, ca, away)):
        sh = [e for e in f["events"] if e[2] == team and e[3] in ("shot", "goal")]
        xs = [0] + [e[1] / 60 for e in sh] + [end / 60]
        ys = list(range(len(sh) + 1)) + [len(sh)]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(shape="hv", color=col, width=3),
                                 name=f"{ab} shots on goal"))
        gl = [(e, i + 1) for i, e in enumerate(sh) if e[3] == "goal"]
        fig.add_trace(go.Scatter(x=[e[1] / 60 for e, _ in gl], y=[c for _, c in gl], mode="markers",
                                 marker=dict(symbol="star", size=16, color=col, line=dict(width=1, color="black")),
                                 name=f"{ab} goals",
                                 hovertext=[f"{clock(e[1])} {event_text(e, f['names'], home, away)}" for e, _ in gl],
                                 hoverinfo="text"))
    for e in f["events"]:
        if e[3] == "pen":
            pp_team = 1 - e[2]
            ends = [g[1] for g in f["events"] if g[3] == "goal" and g[2] == pp_team and e[1] < g[1] <= e[1] + 120]
            stop = min(ends) if ends else e[1] + 120
            fig.add_vrect(x0=e[1] / 60, x1=stop / 60, fillcolor=ch if pp_team == 0 else ca,
                          opacity=0.12, line_width=0)
    for m in (20, 40, 60):
        fig.add_vline(x=m, line=dict(color="#999", dash="dot"))
    fig.update_layout(height=380, xaxis_title="Game minute (shaded = power play)",
                      yaxis_title="Cumulative shots on goal", margin=dict(t=30, b=40),
                      legend=dict(orientation="h", y=1.12))
    return fig


def winprob_figure(res):
    wp, f = res["winprob"], res["featured"]
    home, away = res["home"], res["away"]
    ch, ca = team_colors(home, away)
    t = wp["t"]
    ph = np.array(wp["home"]) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=ph, mode="lines", line=dict(color=ch, width=3),
                             name=f"{home} win %", hovertemplate="%{y:.0f}%<extra>" + home + "</extra>"))
    fig.add_trace(go.Scatter(x=t, y=100 - ph, mode="lines", line=dict(color=ca, width=3),
                             name=f"{away} win %", hovertemplate="%{y:.0f}%<extra>" + away + "</extra>"))
    for e in f["events"]:
        if e[3] == "goal":
            i = max([k for k, tt in enumerate(t) if tt * 60 <= e[1] + DT] or [0])
            y = ph[i] if e[2] == 0 else 100 - ph[i]
            fig.add_trace(go.Scatter(x=[e[1] / 60], y=[y], mode="markers", showlegend=False,
                                     marker=dict(symbol="star", size=15, color=ch if e[2] == 0 else ca,
                                                 line=dict(width=1, color="black")),
                                     hovertext=f"{clock(e[1])} {event_text(e, f['names'], home, away)}",
                                     hoverinfo="text"))
    fig.add_hline(y=50, line=dict(color="#999", dash="dot"))
    for m in (20, 40, 60):
        fig.add_vline(x=m, line=dict(color="#ccc", dash="dot"))
    fig.update_layout(height=340, yaxis=dict(range=[0, 100], title="Win probability %"),
                      xaxis_title="Game minute", margin=dict(t=30, b=40), legend=dict(orientation="h", y=1.12))
    return fig


def american(p):
    p = min(max(p, 0.001), 0.999)
    return f"{-100 * p / (1 - p):.0f}" if p >= 0.5 else f"+{100 * (1 - p) / p:.0f}"


def implied(price):
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def payout(price):
    return price if price > 0 else 10000 / -price


def shot_map(prof, which, color):
    shots = prof["shots_for"] if which == "for" else prof["shots_ag"]
    fig = go.Figure()
    if shots:
        arr = np.array(shots, dtype=float)
        fig.add_trace(go.Histogram2dContour(
            x=arr[:, 0], y=arr[:, 1], colorscale=[[0, "rgba(255,255,255,0)"], [1, color]],
            showscale=False, ncontours=12, contours=dict(coloring="fill", showlines=False),
            xbins=dict(start=0, end=100, size=5), ybins=dict(start=-42.5, end=42.5, size=5), opacity=0.8,
            hoverinfo="skip"))
        gl = arr[arr[:, 3] == 1]
        fig.add_trace(go.Scatter(x=gl[:, 0], y=gl[:, 1], mode="markers", name="goals",
                                 marker=dict(symbol="star", size=10, color="black")))
    lay = rink_layout(330)
    lay["xaxis"]["range"] = [-2, 102]
    lay["margin"] = dict(l=5, r=5, t=10, b=5)
    lay["showlegend"] = False
    fig.update_layout(**lay)
    return fig


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


# --------------------------------------------------------------------------
# Sportsbook odds (the-odds-api.com)
# --------------------------------------------------------------------------
def odds_feed(api_key):
    return [], "?"


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", "", s.replace("-", " ")).strip()


def same_team(feed_name, abbrev):
    full, fn = _norm(NAMES.get(abbrev, abbrev)), _norm(feed_name)
    if not fn or not full:
        return False
    return fn == full or fn.split()[-1] == full.split()[-1] or fn == abbrev.lower()


def secret(name, default=None):
    """A setting from Streamlit secrets, or from an environment variable (used by the daily GitHub robot)."""
    try:
        v = st.secrets.get(name)
        if v:
            return v
    except Exception:
        pass
    return os.environ.get(name) or default


def odds_api_key():
    return ""      # sportsbook lines aren't available for WHL games


def best_lines(evt, res):
    """Best moneyline prices and the main total (with sim chances) for one odds-feed game."""
    ml = {"home": None, "away": None}
    totals = {}
    for bk in evt.get("bookmakers", []):
        for mk in bk.get("markets", []):
            for oc in mk.get("outcomes", []):
                price = oc.get("price")
                if price is None:
                    continue
                if mk["key"] == "h2h":
                    side = "home" if same_team(oc["name"], res["home"]) else "away"
                    if ml[side] is None or price > ml[side]:
                        ml[side] = price
                elif mk["key"] == "totals" and oc.get("point") is not None:
                    t = totals.setdefault(float(oc["point"]), {"n": 0, "Over": None, "Under": None})
                    t["n"] += 1
                    if t[oc["name"]] is None or price > t[oc["name"]]:
                        t[oc["name"]] = price
    out = dict(ml_home=ml["home"], ml_away=ml["away"], total_line=None, over_price=None, under_price=None,
               p_over_line=None, p_under_line=None)
    if totals:
        line = max(totals, key=lambda k: totals[k]["n"])
        tot = res["total"]
        out.update(total_line=line, over_price=totals[line]["Over"], under_price=totals[line]["Under"],
                   p_over_line=round(float((tot > line).mean()), 4), p_under_line=round(float((tot < line).mean()), 4))
    return out


PROP_MARKETS = {"player_shots_on_goal": "SOG", "player_points": "Points", "player_assists": "Assists",
                "player_goals": "Goals", "player_power_play_points": "PP points", "player_total_saves": "Saves",
                "player_goal_scorer_anytime": "Anytime goal"}


@st.cache_data(ttl=900, show_spinner=False)
def prop_feed(api_key, event_id):
    """Player-prop odds for one game (costs about 1 credit per market that has odds)."""
    r = _session().get(f"https://api.the-odds-api.com/v4/sports/icehockey_nhl/events/{event_id}/odds",
                       params=dict(apiKey=api_key, regions="us", markets=",".join(PROP_MARKETS),
                                   oddsFormat="american"), timeout=25)
    r.raise_for_status()
    return r.json(), str(r.headers.get("x-requests-remaining", "?"))


def find_event(feed, home, away):
    return next((e for e in feed or [] if same_team(e.get("home_team"), home)
                 and same_team(e.get("away_team"), away)), None)


def prop_lines(evt_odds):
    """Flatten a prop feed into rows: player, stat, line, side, price, book."""
    rows = []
    for bk in (evt_odds or {}).get("bookmakers", []):
        for mk in bk.get("markets", []):
            stat = PROP_MARKETS.get(mk.get("key"))
            if not stat:
                continue
            for oc in mk.get("outcomes", []):
                name, price = str(oc.get("name", "")), oc.get("price")
                if price is None:
                    continue
                if name in ("Over", "Under", "Yes", "No"):
                    player, side = oc.get("description") or "", name
                else:
                    player, side = name, "Yes"
                if stat == "Anytime goal":
                    if side == "No":
                        continue
                    stat, side, line = "Goals", "Over", 0.5
                else:
                    line = oc.get("point")
                    if line is None:
                        continue
                rows.append(dict(player=player, stat=stat, line=float(line), side=side, price=int(price),
                                 book=bk.get("title", bk.get("key", ""))))
    return rows


# --------------------------------------------------------------------------
# Confirmed starting goalies (DailyFaceoff's public starting-goalies page)
# --------------------------------------------------------------------------
GOALIE_ICON = {"Recent": "🔁 Best guess", "Manual": "✏️ Your pick"}


def match_goalie(name, goalies):
    n = _norm(name)
    for g in goalies:
        if _norm(g["name"]) == n:
            return g["id"]
    return None


def next_game(home, away):
    """The next scheduled game with this home and away team, if there is one."""
    try:
        up = [g for g in season_schedule(season_id()) if g["state"] != "FINAL"
              and g["home"] == home and g["away"] == away]
    except Exception:
        return None
    if not up:
        return None
    g = up[0]
    return dict(id=g["id"], date=g["date"], start=g["start"], state=g["state"], playoff=False)


def starting_goalies(home, away, date_str, hp, ap):
    """Probable starters: recent starts, rest, and who played last night (you can change them in Lineups)."""
    out = {}
    for side, prof in (("home", hp), ("away", ap)):
        gid, why = guess_goalie(prof, date_str)
        out[side] = (gid if gid is not None else default_lineup(prof, False)[0], "Recent", why)
    return out


def goalie_name(prof, gid):
    return next((g["name"] for g in prof["goalies"] if g["id"] == gid), "League-average goalie")


# --------------------------------------------------------------------------
# Prediction tracker (GitHub when set up in Secrets, otherwise a local CSV)
# --------------------------------------------------------------------------
MODEL_VERSION = "whl-v6-2026-11-18"
TRACK_COLS = ["game_id", "date", "start_utc", "away", "home", "away_goalie", "home_goalie", "goalie_status",
              "p_home", "proj_away", "proj_home", "p_ot", "likely", "sims", "n_recent", "model", "source",
              "saved_at", "ml_away", "ml_home", "total_line", "over_price", "under_price", "p_over_line",
              "p_under_line", "final_away", "final_home", "end"]
TRACK_TEXT = ["date", "start_utc", "away", "home", "away_goalie", "home_goalie", "goalie_status", "likely",
              "model", "source", "saved_at", "end"]
PLAYER_COLS = ["game_id", "date", "start_utc", "team", "opp", "player_id", "player", "pos",
               "proj_g", "proj_a", "proj_pts", "proj_sog", "proj_ppp", "proj_saves",
               "p_goal", "p_point", "p_2pts", "p_ast", "p_sog2", "p_sog3", "p_sog4", "p_ppp",
               "model", "saved_at", "act_g", "act_a", "act_sog", "act_ppp", "act_saves", "status"]
PLAYER_TEXT = ["date", "start_utc", "team", "opp", "player", "pos", "model", "saved_at", "status"]
BET_COLS = ["bet_id", "placed_at", "date", "game_id", "start_utc", "game", "market", "selection", "team",
            "player_id", "player", "stat", "side", "line", "odds", "stake", "model_p", "book", "source",
            "status", "profit", "result", "graded_at", "legs"]
BET_TEXT = ["placed_at", "date", "start_utc", "game", "market", "selection", "team", "player", "stat", "side",
            "book", "source", "status", "result", "graded_at", "legs"]
STORES = {
    "games": dict(cols=TRACK_COLS, text=TRACK_TEXT, file="whl_predictions.csv", secret="WHL_TRACKER_PATH"),
    "players": dict(cols=PLAYER_COLS, text=PLAYER_TEXT, file="whl_player_predictions.csv", secret="WHL_PLAYER_TRACKER_PATH"),
    "bets": dict(cols=BET_COLS, text=BET_TEXT, file="whl_my_bets.csv", secret="WHL_BETS_PATH"),
}
APP_DIR = Path(__file__).resolve().parent
LOCAL_TRACK = APP_DIR / STORES["games"]["file"]


def _gh_conf(store="games"):
    token, repo = secret("GITHUB_TOKEN"), secret("GITHUB_REPO")
    if token and repo:
        info = STORES[store]
        return dict(token=token, repo=repo, branch=secret("TRACKER_BRANCH", "tracker-data"),
                    path=secret(info["secret"], info["file"]))
    return None


def _gh(method, url, conf, **kw):
    headers = {"Authorization": f"Bearer {conf['token']}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    headers.update(kw.pop("headers", {}))
    return requests.request(method, f"https://api.github.com{url}", headers=headers, timeout=30, **kw)


def _gh_ensure_branch(conf):
    """Saved data lives on its own branch so saving it doesn't restart the Streamlit app."""
    repo, br = conf["repo"], conf["branch"]
    if _gh("GET", f"/repos/{repo}/git/ref/heads/{br}", conf).status_code == 200:
        return
    r = _gh("GET", f"/repos/{repo}", conf)
    r.raise_for_status()
    base = r.json()["default_branch"]
    r = _gh("GET", f"/repos/{repo}/git/ref/heads/{base}", conf)
    r.raise_for_status()
    _gh("POST", f"/repos/{repo}/git/refs", conf,
        json=dict(ref=f"refs/heads/{br}", sha=r.json()["object"]["sha"])).raise_for_status()


def _read_csv_text(text, cols=TRACK_COLS, text_cols=TRACK_TEXT):
    # text columns are read as text even when empty, so results like "REG" can be written into them later
    df = (pd.read_csv(io.StringIO(text), dtype={c: "object" for c in text_cols}) if text.strip()
          else pd.DataFrame(columns=cols))
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols].copy()
    for c in text_cols:
        df[c] = df[c].astype("object")
    for c in cols:
        if c not in text_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _gh_get(conf):
    """(file text, sha) from GitHub; big files (over 1 MB) are downloaded raw."""
    url = f"/repos/{conf['repo']}/contents/{conf['path']}"
    r = _gh("GET", url, conf, params={"ref": conf["branch"]})
    if r.status_code == 404:
        return "", None
    r.raise_for_status()
    js = r.json()
    if js.get("content") and js.get("encoding") == "base64":
        return base64.b64decode(js["content"]).decode(), js.get("sha")
    raw = _gh("GET", url, conf, params={"ref": conf["branch"]}, headers={"Accept": "application/vnd.github.raw+json"})
    raw.raise_for_status()
    return raw.text, js.get("sha")


@st.cache_data(ttl=120, show_spinner=False)
def load_store(store):
    info = STORES[store]
    conf = _gh_conf(store)
    if conf:
        text = _gh_get(conf)[0]
    else:
        f = APP_DIR / info["file"]
        text = f.read_text() if f.exists() else ""
    return _read_csv_text(text, info["cols"], info["text"])


def save_store(store, df, message):
    info = STORES[store]
    text = df[info["cols"]].to_csv(index=False)
    conf = _gh_conf(store)
    if conf:
        _gh_ensure_branch(conf)
        sha = _gh_get(conf)[1]
        body = dict(message=message, branch=conf["branch"], content=base64.b64encode(text.encode()).decode())
        if sha:
            body["sha"] = sha
        _gh("PUT", f"/repos/{conf['repo']}/contents/{conf['path']}", conf, json=body).raise_for_status()
    else:
        (APP_DIR / info["file"]).write_text(text)
    load_store.clear()


def load_tracker():
    return load_store("games")


def save_tracker(df, message):
    save_store("games", df, message)


def make_record(res, game, goalies, n_recent, source, evt=None):
    s = res["top_scores"][0][0]
    rec = dict(
        game_id=int(game["id"]), date=game.get("date", ""), start_utc=game.get("start", ""),
        away=res["away"], home=res["home"], away_goalie=goalies["away"][0], home_goalie=goalies["home"][0],
        goalie_status=f"{goalies['away'][1]}/{goalies['home'][1]}",
        p_home=round(float(res["home_win"]), 4), proj_away=round(float(res["goals_mean"][1]), 2),
        proj_home=round(float(res["goals_mean"][0]), 2), p_ot=round(float(res["ot"] + res["so"]), 4),
        likely=f"{s[1]}-{s[0]}", sims=int(res["n"]), n_recent=int(n_recent), model=MODEL_VERSION, source=source,
        saved_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        final_away=None, final_home=None, end=None)
    rec.update(best_lines(evt, res) if evt else dict(ml_away=None, ml_home=None, total_line=None, over_price=None,
                                                    under_price=None, p_over_line=None, p_under_line=None))
    return rec


def _started(start_utc):
    try:
        return pd.Timestamp(start_utc) <= pd.Timestamp.now(tz="UTC")
    except Exception:
        return False


ODDS_COLS = ["ml_away", "ml_home", "total_line", "over_price", "under_price", "p_over_line", "p_under_line"]


def upsert_predictions(records, message):
    """Add or refresh predictions. Once a game has started its prediction is locked."""
    df = load_tracker().copy()
    changed = 0
    for rec in records:
        if _started(rec["start_utc"]):
            continue
        hit = df.index[pd.to_numeric(df["game_id"], errors="coerce") == rec["game_id"]]
        if len(hit):
            row = df.loc[hit[0]]
            if pd.notna(row["final_home"]) or _started(row["start_utc"]):
                continue
            if pd.isna(rec.get("ml_home")) and pd.notna(row["ml_home"]):     # keep odds saved by an earlier run
                rec = dict(rec, **{c: row[c] for c in ODDS_COLS})
            df = df.drop(hit)
        df = pd.concat([df, pd.DataFrame([rec])[TRACK_COLS]], ignore_index=True)
        changed += 1
    if changed:
        save_tracker(df.sort_values(["date", "start_utc", "game_id"]), message)
    return changed


def grade_tracker():
    """Fill in final scores for finished games; saves only if something changed."""
    df = load_tracker().copy()
    todo = df[df["final_home"].isna() & df["start_utc"].apply(_started)]
    changed = 0
    for date_str, grp in todo.groupby("date"):
        scores = final_scores(str(date_str))
        for i, row in grp.iterrows():
            gid = int(row["game_id"])
            sc = scores.get(gid)
            if sc and sc[0] is not None:
                df.loc[i, "final_away"] = float(sc[0])
                df.loc[i, "final_home"] = float(sc[1])
                df.loc[i, "end"] = str(sc[2])
                changed += 1
    if changed:
        save_tracker(df, f"Grade {changed} WHL predictions")
    return df, changed


def bet_results(df, min_ev):
    """Flat $100 on every bet whose expected profit per $100 is at least min_ev."""
    bets = []
    for _, r in df.iterrows():
        if pd.isna(r["final_home"]):
            continue
        fh, fa = float(r["final_home"]), float(r["final_away"])
        p = float(r["p_home"])
        for side, prob, price, won in (("home", p, r["ml_home"], fh > fa), ("away", 1 - p, r["ml_away"], fa > fh)):
            if pd.isna(price):
                continue
            ev = prob * payout(price) - (1 - prob) * 100
            if ev >= min_ev:
                team = r["home"] if side == "home" else r["away"]
                bets.append(dict(date=r["date"], game=f"{r['away']} @ {r['home']}", bet=f"{team} ML {int(price):+d}",
                                 ev=ev, profit=payout(price) if won else -100.0))
        if pd.notna(r["total_line"]) and pd.notna(r["p_over_line"]):
            tot, line = fh + fa, float(r["total_line"])
            p_o, p_u = float(r["p_over_line"]), float(r["p_under_line"])
            for name, prob, lose, price, won, push in (
                    ("Over", p_o, p_u, r["over_price"], tot > line, tot == line),
                    ("Under", p_u, p_o, r["under_price"], tot < line, tot == line)):
                if pd.isna(price):
                    continue
                ev = prob * payout(price) - lose * 100     # a push (whole-number line) neither wins nor loses
                if ev >= min_ev:
                    bets.append(dict(date=r["date"], game=f"{r['away']} @ {r['home']}",
                                     bet=f"{name} {line:g} {int(price):+d}", ev=ev,
                                     profit=0.0 if push else (payout(price) if won else -100.0)))
    return pd.DataFrame(bets, columns=["date", "game", "bet", "ev", "profit"])




# --------------------------------------------------------------------------
# Live scores, box scores and player-level results
# --------------------------------------------------------------------------
def game_clock(g):
    """(seconds elapsed, label) for a live game from the scoreboard feed."""
    per = int((g.get("periodDescriptor") or {}).get("number") or g.get("period") or 1)
    ck = g.get("clock") or {}
    rem = ck.get("secondsRemaining")
    if rem is None:
        rem = _secs(ck.get("timeRemaining", "20:00"))
    length = 1200 if per <= 3 or g.get("gameType") == 3 else 300
    elapsed = (per - 1) * 1200 + (length - int(rem))
    if ck.get("inIntermission"):
        label = f"End of {['1st', '2nd', '3rd'][per - 1] if per <= 3 else 'OT'}"
    else:
        name = f"P{per}" if per <= 3 else ("OT" if per == 4 else f"{per - 3}OT")
        label = f"{name} {int(rem) // 60:02d}:{int(rem) % 60:02d}"
    return elapsed, label


STAT_KEYS = {"SOG": "SOG", "Goals": "G", "Assists": "A", "PP points": "PPP"}


def stat_array(block, idx, stat):
    """Simulated values of one stat for one player (index idx in a prop block)."""
    if stat == "Points":
        return block["G"][:, idx].astype(np.int16) + block["A"][:, idx]
    return block[STAT_KEYS[stat]][:, idx]


def find_player(res, pid=None, name=None):
    """(block, index) for a player in a simulated game, by id or by name."""
    for b in res.get("props", []):
        sid = [str(x) for x in b["ids"]]
        if pid is not None and str(pid) in sid:
            return b, sid.index(str(pid))
    if name:
        n = _norm(name)
        for b in res.get("props", []):
            for i, nm in enumerate(b["names"]):
                if _norm(nm) == n:
                    return b, i
        parts = n.split()
        if parts:
            hits = [(b, i) for b in res.get("props", []) for i, nm in enumerate(b["names"])
                    if _norm(nm).split()[-1:] == parts[-1:] and _norm(nm)[:1] == n[:1]]
            if len(hits) == 1:
                return hits[0]
    return None, None


def find_goalie(res, pid=None, name=None):
    for b in res.get("props", []):
        if (pid is not None and str(b.get("goalie_id")) == str(pid)) or (name and _norm(b["goalie"]) == _norm(name)):
            return b
    if name:
        last = _norm(name).split()[-1:]
        hits = [b for b in res.get("props", []) if _norm(b["goalie"]).split()[-1:] == last]
        if len(hits) == 1:
            return hits[0]
    return None


def over_under(arr, line, side):
    """(win, push) chance of an Over/Under bet on simulated values."""
    arr = np.asarray(arr)
    if side in ("Over", "Yes"):
        return float((arr > line).mean()), float((arr == line).mean())
    return float((arr < line).mean()), float((arr == line).mean())


def model_prob(res, market, team=None, side=None, line=None, player_id=None, stat=None, player=None):
    """(win, push) chance for one bet according to the simulation, or (None, None) if it can't be priced."""
    try:
        m, tot = res["margin"], res["total"]
        if market == "Moneyline":
            return float(((m > 0) if team == res["home"] else (m < 0)).mean()), 0.0
        if market == "Puck line":
            tm = m if team == res["home"] else -m
            return float((tm + line > 0).mean()), float((tm + line == 0).mean())
        if market == "Total":
            return over_under(tot, float(line), side)
        if market == "Player":
            pid = None if player_id is None or pd.isna(player_id) else int(player_id)
            if stat == "Saves":
                b = find_goalie(res, pid, player)
                return over_under(b["saves"], float(line), side) if b is not None else (None, None)
            b, i = find_player(res, pid, player)
            if b is None:
                return None, None
            return over_under(stat_array(b, i, stat), float(line), side)
    except Exception:
        pass
    return None, None


def leg_outcome(res, leg):
    """(win, push) boolean arrays over every simulated game for one bet leg (None if it can't be priced)."""
    m, tot = res["margin"].astype(int), res["total"].astype(int)
    mk, side, line = leg["market"], leg.get("side"), leg.get("line")
    line = None if line is None or (isinstance(line, float) and np.isnan(line)) else float(line)

    def ou(arr):
        arr = np.asarray(arr).astype(float)
        if side in ("Over", "Yes"):
            return arr > line, arr == line
        return arr < line, arr == line

    if mk == "Moneyline":
        return ((m > 0) if leg["team"] == res["home"] else (m < 0)), np.zeros(len(m), bool)
    if mk == "Puck line":
        tm = m if leg["team"] == res["home"] else -m
        return tm + line > 0, tm + line == 0
    if mk == "Total":
        return ou(tot)
    if mk == "Player":
        pid = leg.get("player_id")
        pid = None if pid is None or (isinstance(pid, float) and np.isnan(pid)) else int(pid)
        if leg["stat"] == "Saves":
            b = find_goalie(res, pid, leg.get("player"))
            return ou(b["saves"]) if b is not None else None
        b, i = find_player(res, pid, leg.get("player"))
        return ou(stat_array(b, i, leg["stat"])) if b is not None else None
    return None


def parlay_prob(res, legs):
    """Joint chance that every leg wins in the SAME simulated game (so legs that move together are
    priced correctly), the chance if the legs were unrelated, and the push/void chance."""
    outs = [leg_outcome(res, l) for l in legs]
    if not outs or any(o is None for o in outs):
        return None
    win = np.logical_and.reduce([o[0] for o in outs])
    lose_any = np.logical_or.reduce([~o[0] & ~o[1] for o in outs])
    indep = float(np.prod([o[0].mean() for o in outs]))
    return dict(joint=float(win.mean()), indep=indep, push=float((~win & ~lose_any).mean()),
                legs=[float(o[0].mean()) for o in outs])


def kelly(p, price):
    """Full-Kelly fraction of bankroll (0 if the bet has no edge)."""
    if p is None or price is None:
        return 0.0
    b = payout(price) / 100
    return max(0.0, (p * b - (1 - p)) / b)


# ---------------- Player prediction tracker ----------------
def player_records(res, game, top=12):
    """Rows for the player tracker: the top skaters on each team by projected points, plus both goalies."""
    rows = []
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for k, b in enumerate(res["props"]):
        opp = res["props"][1 - k]["team"]
        pts = b["G"].astype(np.int16) + b["A"]
        order = np.argsort(-pts.mean(0))[:top]
        for i in order:
            g_, a_, s_, p_ = b["G"][:, i], b["A"][:, i], b["SOG"][:, i], b["PPP"][:, i]
            rows.append(dict(game_id=int(game["id"]), date=game.get("date", ""), start_utc=game.get("start", ""),
                             team=b["team"], opp=opp, player_id=b["ids"][i], player=b["names"][i], pos=b["pos"][i],
                             proj_g=round(float(g_.mean()), 3), proj_a=round(float(a_.mean()), 3),
                             proj_pts=round(float(pts[:, i].mean()), 3), proj_sog=round(float(s_.mean()), 2),
                             proj_ppp=round(float(p_.mean()), 3), proj_saves=None,
                             p_goal=round(float((g_ > 0).mean()), 4), p_point=round(float((pts[:, i] > 0).mean()), 4),
                             p_2pts=round(float((pts[:, i] > 1).mean()), 4), p_ast=round(float((a_ > 0).mean()), 4),
                             p_sog2=round(float((s_ >= 2).mean()), 4), p_sog3=round(float((s_ >= 3).mean()), 4),
                             p_sog4=round(float((s_ >= 4).mean()), 4), p_ppp=round(float((p_ > 0).mean()), 4),
                             model=MODEL_VERSION, saved_at=now, status=None))
        if b.get("goalie_id"):
            rows.append(dict(game_id=int(game["id"]), date=game.get("date", ""), start_utc=game.get("start", ""),
                             team=b["team"], opp=opp, player_id=b["goalie_id"], player=b["goalie"], pos="G",
                             proj_saves=round(float(b["saves"].mean()), 2), model=MODEL_VERSION, saved_at=now,
                             status=None))
    return rows


def upsert_players(records, message):
    """Replace a game's player predictions until it starts; finished or started games stay locked."""
    if not records:
        return 0
    df = load_store("players").copy()
    gids = {r["game_id"] for r in records if not _started(r["start_utc"])}
    if not gids:
        return 0
    have = pd.to_numeric(df["game_id"], errors="coerce")
    df = df[~have.isin(gids)]
    new = pd.DataFrame([r for r in records if r["game_id"] in gids])
    for c in PLAYER_COLS:
        if c not in new.columns:
            new[c] = None
    df = pd.concat([df, new[PLAYER_COLS]], ignore_index=True)
    save_store("players", df.sort_values(["date", "game_id", "team"]), message)
    return len(gids)


def grade_players():
    """Fill in real stat lines for finished games (saves only if something changed)."""
    df = load_store("players").copy()
    todo = df[df["status"].isna() & df["start_utc"].apply(_started)]
    changed = 0
    for gid, grp in todo.groupby("game_id"):
        done, box = boxscore_stats(int(gid))
        if not done:
            continue
        ppp = pp_points(str(grp["date"].iloc[0])).get(int(gid), Counter())
        for i, r in grp.iterrows():
            st_ = box.get(int(r["player_id"]))
            if st_ is None:
                df.loc[i, "status"] = "dnp"
            elif r["pos"] == "G":
                if st_.get("toi") in ("", "00:00", None):
                    df.loc[i, "status"] = "dnp"
                else:
                    df.loc[i, "act_saves"] = float(st_["saves"])
                    df.loc[i, "status"] = "graded"
            else:
                df.loc[i, "act_g"] = float(st_["G"])
                df.loc[i, "act_a"] = float(st_["A"])
                df.loc[i, "act_sog"] = float(st_["SOG"])
                df.loc[i, "act_ppp"] = float(ppp.get(int(r["player_id"]), st_.get("PPG", 0)))
                df.loc[i, "status"] = "graded"
            changed += 1
    if changed:
        save_store("players", df, f"Grade {changed} WHL player predictions")
    return df, changed


# ---------------- My bets ----------------
def add_bets(rows):
    df = load_store("bets").copy()
    now = dt.datetime.now(dt.timezone.utc)
    new = []
    for k, r in enumerate(rows):
        r = dict(r)
        r.setdefault("bet_id", int(now.timestamp() * 1000) + k)
        r.setdefault("placed_at", now.strftime("%Y-%m-%dT%H:%M:%SZ"))
        r.setdefault("status", "Open")
        new.append(r)
    nd = pd.DataFrame(new)
    for c in BET_COLS:
        if c not in nd.columns:
            nd[c] = None
    df = pd.concat([df, nd[BET_COLS]], ignore_index=True)
    save_store("bets", df, f"Log {len(new)} bet(s)")
    return len(new)


def bet_label(r):
    if r["market"] == "Parlay":
        return " + ".join(bet_label(l) for l in parlay_legs(r))
    if r["market"] == "Moneyline":
        return f"{r['team']} moneyline"
    if r["market"] == "Puck line":
        return f"{r['team']} {float(r['line']):+g}"
    if r["market"] == "Total":
        return f"{r['side']} {float(r['line']):g} goals"
    return f"{r['player']} {r['side']} {float(r['line']):g} {r['stat']}"


def parlay_legs(r):
    try:
        legs = json.loads(r["legs"]) if isinstance(r.get("legs"), str) and r["legs"] else []
    except Exception:
        legs = []
    for l in legs:
        l.setdefault("game", r.get("game"))
        for k in ("team", "side", "line", "player_id", "player", "stat"):
            l.setdefault(k, None)
    return legs


def settle(r, final, box, ppp):
    """('Won'|'Lost'|'Push'|'Void', note) for one bet, given the final score and box score."""
    if r["market"] == "Parlay":
        res_ = [settle(l, final, box, ppp) for l in parlay_legs(r)]
        stats = [x[0] for x in res_]
        note = " · ".join(f"{'✅' if s_ == 'Won' else '❌' if s_ == 'Lost' else '➖'} {n}" for s_, n in res_)
        if not stats:
            return "Void", "No legs saved"
        if "Lost" in stats:
            return "Lost", note
        if all(s_ == "Won" for s_ in stats):
            return "Won", note
        return "Void", note + " (a leg pushed or was void: books re-price these, so check your payout)"
    fa, fh, end = final
    margin, tot = fh - fa, fh + fa
    mk, side, line = r["market"], r["side"], (None if pd.isna(r["line"]) else float(r["line"]))
    if mk == "Moneyline":
        won = (margin > 0) == (r["team"] == r_home(r))
        return ("Won" if won else "Lost"), f"Final {fa}-{fh}" + (f" ({end})" if end in ("OT", "SO") else "")
    if mk == "Puck line":
        tm = margin if r["team"] == r_home(r) else -margin
        v = tm + line
        return ("Won" if v > 0 else "Push" if v == 0 else "Lost"), f"Final {fa}-{fh}"
    if mk == "Total":
        v, note = tot, f"{tot} goals"
    else:
        pid = None if pd.isna(r["player_id"]) else int(r["player_id"])
        s_ = box.get(pid)
        if s_ is None or (r["stat"] == "Saves" and s_.get("toi") in ("", "00:00", None)):
            return "Void", "Did not play (stake back)"
        v = {"SOG": s_.get("SOG", 0), "Goals": s_.get("G", 0), "Assists": s_.get("A", 0),
             "Points": s_.get("G", 0) + s_.get("A", 0), "Saves": s_.get("saves", 0),
             "PP points": ppp.get(pid, s_.get("PPG", 0))}.get(r["stat"], 0)
        note = f"{v} {r['stat']}"
    if v == line:
        return "Push", note
    over = v > line
    return ("Won" if over == (side in ("Over", "Yes")) else "Lost"), note


def r_home(r):
    return str(r["game"]).split(" @ ")[-1].strip()


def grade_bets():
    df = load_store("bets").copy()
    todo = df[(df["status"] == "Open") & df["start_utc"].apply(_started)]
    changed = 0
    for gid, grp in todo.groupby("game_id"):
        gid = int(gid)
        done, box = boxscore_stats(gid)
        if not done:
            continue
        date_str = str(grp["date"].iloc[0])
        sc = final_scores(date_str).get(gid)
        if sc is None:
            continue
        ppp = pp_points(date_str).get(gid, Counter())
        for i, r in grp.iterrows():
            status, note = settle(r, (int(sc[0]), int(sc[1]), str(sc[2])), box, ppp)
            stake = float(r["stake"] or 0)
            df.loc[i, "status"] = status
            df.loc[i, "result"] = note
            df.loc[i, "profit"] = stake * payout(r["odds"]) / 100 if status == "Won" else (-stake if status == "Lost" else 0.0)
            df.loc[i, "graded_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            changed += 1
    if changed:
        save_store("bets", df, f"Grade {changed} bet(s)")
    return df, changed



# --------------------------------------------------------------------------
# Backtest: replay this season's finished games using only what was known before each one
# --------------------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def season_games():
    """[(game id, date, home, away)] for every finished regular-season game this season."""
    try:
        games = season_schedule(season_id())
    except Exception:
        return []
    return [(g["id"], g["date"], g["home"], g["away"]) for g in games if g["state"] == "FINAL"]


def game_starters(g):
    """{team id: starting goalie} for one slimmed game (first goalie to face a shot)."""
    out = {}
    for (t, per, typ, owner, *_rest, goalie) in g["plays"]:
        if typ in ("shot-on-goal", "goal") and goalie:
            other = g["away_id"] if owner == g["home_id"] else g["home_id"]
            out.setdefault(other, goalie)
        if len(out) == 2:
            break
    return out


def asof_players(team, prior):
    """Skater and goalie stats rebuilt from earlier game files only (nothing from later games)."""
    sk, gk = {}, {}
    for g in prior:
        tid = g["home_id"] if g["home"] == team else g["away_id"]
        info = g.get("rinfo") or {}
        st_ = g.get("pstats") or {}
        for pid, t_ in g["roster"]:
            if t_ != tid:
                continue
            pos, name = info.get(pid, ("F", str(pid)))
            if pos == "G":
                gk.setdefault(pid, dict(id=pid, name=name, gp=0, gs=0, sa=0, ga=0))
                continue
            d = sk.setdefault(pid, dict(id=pid, name=name, pos=pos, gp=0, g=0, a=0, shots=0, ppp=0, pim=0))
            d["gp"] += 1
            v = st_.get(pid, [0, 0, 0, 0])
            d["shots"] += v[0]
            d["g"] += v[1]
            d["a"] += v[2]
            d["ppp"] += v[3]
        starters = game_starters(g)
        if tid in starters and starters[tid] in gk:
            gk[starters[tid]]["gs"] += 1
        played = set()
        for (t, per, typ, owner, *_rest, goalie) in g["plays"]:
            if typ in ("shot-on-goal", "goal") and owner != tid and goalie in gk:
                gk[goalie]["sa"] += 1
                gk[goalie]["ga"] += typ == "goal"
                played.add(goalie)
        for gid in played:
            gk[gid]["gp"] += 1
    skaters = []
    for d in sk.values():
        gp = max(d["gp"], 1)
        d["toi"] = float(700 + 260 * (d["g"] + d["a"]) / gp + 110 * d["shots"] / gp + (420 if d["pos"] == "D" else 0))
        skaters.append(d)
    goalies = []
    for d in gk.values():
        d["saves"] = d["sa"] - d["ga"]
        d["svp"] = d["saves"] / d["sa"] if d["sa"] else 0.0
        goalies.append(d)
    return skaters, goalies


def _bt_context():
    games = season_games()
    pb = {g["id"]: g for g in fetch_pbps([r[0] for r in games])}
    return games, pb


def asof_profile(team, date, n_recent, games, pb):
    prior = [pb[gid] for gid, d, h, a in games if d < date and team in (h, a) and gid in pb]
    if not prior:
        return None, 0
    prof = _profile_core(team, prior[-n_recent:])
    prof["skaters"], prof["goalies"] = asof_players(team, prior)
    prof["asof"] = True
    return prof, len(prior)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def backtest_games(n_recent, min_prior=5, reps=800):
    """Game-level backtest: for every finished game, the win chance and total-goals projection the model
    would have made that morning (team style from earlier games only, the goalie who actually started)."""
    games, pb = _bt_context()
    meta, vecs_h, vecs_a, b2b = [], [], [], []
    for gid, date, home, away in games:
        g = pb.get(gid)
        if g is None or g.get("home_score") is None:
            continue
        hp, nh = asof_profile(home, date, n_recent, games, pb)
        ap, na = asof_profile(away, date, n_recent, games, pb)
        if nh < min_prior or na < min_prior:
            continue
        shots = hp["shots_for"] + hp["shots_ag"] + ap["shots_for"] + ap["shots_ag"]
        model = dist_model(np.array(shots, float)[:, 2], np.array(shots, float)[:, 3]) if shots else DIST_PRIOR.copy()
        stt = game_starters(g)
        vecs_h.append(team_vector(hp, model, stt.get(g["home_id"])))
        vecs_a.append(team_vector(ap, model, stt.get(g["away_id"])))
        b2b.append((fatigue_mult(home, date, home), fatigue_mult(away, date, home)))
        meta.append(dict(game_id=gid, date=date, home=home, away=away, hs=int(g["home_score"]), as_=int(g["away_score"]),
                         end="OT/SO" if any(p[1] >= 4 for p in g["plays"]) else "REG"))
    if not meta:
        return pd.DataFrame()
    m = len(meta)
    idx = np.repeat(np.arange(m), reps)
    S = pair_sides(stack(vecs_h, idx), stack(vecs_a, idx))
    hf, hag = (np.array([b[0][j] for b in b2b])[idx] for j in (0, 1))
    af, aag = (np.array([b[1][j] for b in b2b])[idx] for j in (0, 1))
    for key in ("r_ev", "r_pp", "r_sh"):
        f0 = hf * aag
        f1 = af * hag
        S[0][key] = S[0][key] * (f0[:, None] if S[0][key].ndim == 2 else f0)
        S[1][key] = S[1][key] * (f1[:, None] if S[1][key].ndim == 2 else f1)
    gl = fast_reg(S, seed=11)
    d = gl[:, 0] - gl[:, 1]
    tie = d == 0
    p_ot = ot_home_prob(S, False)
    win = (d > 0) + tie * p_ot
    tot = gl.sum(1) + tie            # overtime / shootout adds one goal for the winner
    df = pd.DataFrame(meta)
    df["p_home"] = win.reshape(m, reps).mean(1)
    df["proj_total"] = tot.reshape(m, reps).mean(1)
    df["p_over55"] = (tot > 5.5).reshape(m, reps).mean(1)
    df["p_over65"] = (tot > 6.5).reshape(m, reps).mean(1)
    df["p_ot"] = tie.reshape(m, reps).mean(1)
    df["home_won"] = (df["hs"] > df["as_"]).astype(int)
    df["total"] = df["hs"] + df["as_"]
    return df


@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=2000)
def backtest_players(gid, n_recent, n=600):
    """Player-level backtest for one game: the full simulation with the lineups that actually dressed."""
    games, pb = _bt_context()
    row = next((r for r in games if r[0] == gid), None)
    g = pb.get(gid)
    if row is None or g is None:
        return []
    _, date, home, away = row
    hp, nh = asof_profile(home, date, n_recent, games, pb)
    ap, na = asof_profile(away, date, n_recent, games, pb)
    if nh < 5 or na < 5:
        return []
    shots = hp["shots_for"] + hp["shots_ag"] + ap["shots_for"] + ap["shots_ag"]
    model = dist_model(np.array(shots, float)[:, 2], np.array(shots, float)[:, 3]) if shots else DIST_PRIOR.copy()
    stt = game_starters(g)
    sides = []
    for prof, opp, tid in ((hp, ap, g["home_id"]), (ap, hp, g["away_id"])):
        dressed = {pid for pid, t_ in g["roster"] if t_ == tid}
        out = tuple(s_["id"] for s_ in prof["skaters"] if s_["id"] not in dressed)
        sides.append(build_side(prof, opp, stt.get(tid), out, model))
    H, A = sides
    for side, mult in ((H, 1 + HOME_EDGE), (A, 1 - HOME_EDGE)):
        for key in ("r_ev", "r_pp", "r_sh"):
            side[key] = side[key] * mult
    r = simulate(H, A, n, seed=gid % 997, record=1)
    actual = g.get("pstats") or {}
    out_rows = []
    for k, (side, prof) in enumerate(((H, hp), (A, ap))):
        prior = {s_["id"]: s_ for s_ in prof["skaters"]}
        for i, pid in enumerate(side["ids"]):
            if pid not in {p for p, _ in g["roster"]}:
                continue
            G, A_, S_ = r["g_mat"][k][:, i], r["a_mat"][k][:, i], r["s_mat"][k][:, i]
            pts = G + A_
            v = actual.get(pid, [0, 0, 0, 0])
            ps = prior.get(pid, {})
            gp = max(ps.get("gp", 0), 1)
            base_pt = 1 - math.exp(-(ps.get("g", 0) + ps.get("a", 0) + 0.4) / (gp + 1))   # simple season-average guess
            out_rows.append(dict(game_id=gid, date=date, team=side["team"], player=side["names"][i],
                                 p_point=float((pts > 0).mean()), p_goal=float((G > 0).mean()),
                                 p_sog3=float((S_ >= 3).mean()), proj_sog=float(S_.mean()), proj_pts=float(pts.mean()),
                                 base_point=base_pt, act_sog=v[0], act_g=v[1], act_a=v[2]))
    return out_rows


# --------------------------------------------------------------------------
# WHL live scoreboard, game sheets (grading), player game logs, logos
# --------------------------------------------------------------------------
def logo_url(abbr, dark=False):
    return (team_logo(abbr) or "").replace("/50x50/", "/")


def _iso_to_utc(v):
    try:
        return pd.Timestamp(v).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""


@st.cache_data(ttl=45, show_spinner=False)
def scoreboard(date_str):
    """Every WHL game on a date, with live score / period / clock (refreshes every 45 seconds)."""
    out = []
    try:
        d = dt.date.fromisoformat(str(date_str))
        today = dt.date.today()
        back, ahead = max(0, (today - d).days) + 1, max(0, (d - today).days) + 1
        js = ht_get(dict(feed="modulekit", view="scorebar", numberofdaysback=back, numberofdaysahead=ahead,
                         league_id=HT_LEAGUE_ID, site_id=HT_SITE_ID, lang="en", fmt="json"))
        rows = (js.get("SiteKit") or {}).get("Scorebar") or [] if isinstance(js, dict) else []
    except Exception:
        rows = []
    for r in rows:
        if str(r.get("Date", ""))[:10] != str(date_str):
            continue
        status, text = str(r.get("GameStatus", "")), str(r.get("GameStatusStringLong") or r.get("GameStatusString") or "")
        if "postpon" in text.lower() or "cancel" in text.lower():
            continue
        state = "OFF" if status == "4" or text.lower().startswith("final") else ("FUT" if status in ("1", "") else "LIVE")
        low = text.lower()
        out.append(dict(
            id=int(r["ID"]), gameType=2, gameState=state, startTimeUTC=_iso_to_utc(r.get("GameDateISO8601")),
            homeTeam=dict(abbrev=str(r.get("HomeCode", "")).upper(),
                          score=int(_num(r.get("HomeGoals"))) if state != "FUT" else None),
            awayTeam=dict(abbrev=str(r.get("VisitorCode", "")).upper(),
                          score=int(_num(r.get("VisitorGoals"))) if state != "FUT" else None),
            periodDescriptor=dict(number=int(_num(r.get("Period"), 1)) or 1),
            clock=dict(timeRemaining=str(r.get("GameClock") or "20:00"), inIntermission=str(r.get("Intermission")) == "1"),
            gameOutcome=dict(lastPeriodType="SO" if " so" in f" {low}" else ("OT" if " ot" in f" {low}" else "REG"))))
    if not out:       # fall back to the season schedule
        try:
            for g in season_schedule(season_id()):
                if g["date"] == str(date_str):
                    fin = g["state"] == "FINAL"
                    out.append(dict(id=g["id"], gameType=2, gameState="OFF" if fin else ("LIVE" if g["state"] == "LIVE" else "FUT"),
                                    startTimeUTC=g["start"], homeTeam=dict(abbrev=g["home"], score=g["hs"] if fin else None),
                                    awayTeam=dict(abbrev=g["away"], score=g["as_"] if fin else None),
                                    gameOutcome=dict(lastPeriodType=g["end"])))
        except Exception:
            pass
    return sorted(out, key=lambda g: (g.get("startTimeUTC", ""), g["id"]))


@st.cache_data(ttl=600, show_spinner=False)
def final_scores(date_str):
    """{game id: (away goals, home goals, 'REG'|'OT'|'SO')} for finished games on a date."""
    out = {g["id"]: (g["awayTeam"]["score"], g["homeTeam"]["score"], g["gameOutcome"]["lastPeriodType"])
           for g in scoreboard(date_str) if g["gameState"] == "OFF"}
    if not out:
        try:
            out = {g["id"]: (g["as_"], g["hs"], g["end"]) for g in all_games()
                   if g["date"] == str(date_str) and g["state"] == "FINAL"}
        except Exception:
            pass
    return out


@st.cache_data(ttl=600, show_spinner=False)
def game_sheet(gid):
    try:
        d = ht_get(dict(feed="gc", tab="gamesummary", game_id=gid, fmt="json", lang_code="en", league_id=HT_LEAGUE_ID))
        return (d.get("GC") or {}).get("Gamesummary") or {} if isinstance(d, dict) else {}
    except Exception:
        return {}


def boxscore_stats(gid):
    """(finished?, {player id: stat line}) from the WHL game sheet."""
    gs = game_sheet(int(gid))
    meta = gs.get("meta") or {}
    done = str(meta.get("status")) == "4" or str(meta.get("final")) == "1" or \
        str(gs.get("status_value", "")).lower().startswith("final")
    out = {}
    for key in ("home_team_lineup", "visitor_team_lineup"):
        lu = gs.get(key) or {}
        for p in lu.get("players") or []:
            try:
                pid = int(p.get("player_id"))
            except (TypeError, ValueError):
                continue
            out[pid] = dict(G=int(_num(p.get("goals"))), A=int(_num(p.get("assists"))),
                            SOG=int(_num(p.get("shots_on") or p.get("shots"))), PPG=int(_num(p.get("power_play_goals"))),
                            toi="")
        for p in lu.get("goalies") or []:
            try:
                pid = int(p.get("player_id"))
            except (TypeError, ValueError):
                continue
            played = _num(p.get("seconds")) > 0
            sa_, ga_ = _num(p.get("shots_against")), _num(p.get("goals_against"))
            out[pid] = dict(saves=int(_num(p.get("saves"), sa_ - ga_)), toi=str(p.get("time") or "60:00") if played else "",
                            G=0, A=int(_num(p.get("assists"))), SOG=0)
    return done, out


def pp_points(date_str):
    """{game id: Counter(player id -> power-play points)} for finished games, from the game sheets."""
    out = {}
    for g in scoreboard(date_str):
        if g["gameState"] != "OFF":
            continue
        c = Counter()
        for gl in game_sheet(g["id"]).get("goals") or []:
            if str(gl.get("power_play")) != "1":
                continue
            for k in ("goal_scorer", "assist1_player", "assist2_player"):
                pid = (gl.get(k) or {}).get("player_id")
                if pid:
                    c[int(pid)] += 1
        out[g["id"]] = c
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def _player_log(pid):
    try:
        js = ht_get(dict(feed="statviewfeed", view="player", player_id=pid, season_id=season_id(), site_id=HT_SITE_ID,
                         league_id=HT_LEAGUE_ID, lang="en", statsType="standard"))
    except Exception:
        return []
    rows = []
    for sec in _sections(js.get("gameByGame") if isinstance(js, dict) else None):
        for item in sec.get("data", []):
            rows.append(item.get("row", item))
    return sorted(rows, key=lambda r: str(r.get("date_played", "")), reverse=True)


def recent_values(pid, stat, n=10, team=None):
    """[(date, opponent, value)] from the player's real WHL games (newest first)."""
    if stat == "SOG" and team:       # shots per game only exist in the team's game sheets
        out = []
        try:
            games = fetch_pbps(recent_game_ids(team, 15))
        except Exception:
            games = []
        for g in reversed(games):
            if str(pid) in {str(p) for p, _ in g.get("roster", [])}:
                opp = ("vs " + g["away"]) if g["home"] == team else ("@ " + g["home"])
                out.append((g.get("date", ""), opp, int((g.get("pshots") or {}).get(str(pid), 0))))
            if len(out) >= n:
                break
        return out
    if stat in ("SOG", "PP points"):
        return []
    out = []
    for r in _player_log(pid)[:n]:
        if stat == "Saves":
            if r.get("saves") in (None, "") and r.get("shots_against") in (None, ""):
                continue
            v = _num(r.get("saves"), _num(r.get("shots_against")) - _num(r.get("goals_against")))
        else:
            v = _num(r.get({"Goals": "goals", "Assists": "assists", "Points": "points"}[stat]))
        out.append((str(r.get("date_played", "")), str(r.get("game", "")), int(v)))
    return out


# ==========================================================================
# WHL extras: real team colours, arenas & bus trips, fatigue, smarter goalie guess
# ==========================================================================
# keyword in the team's name -> (primary colour, secondary colour, arena latitude, longitude)
# Colours: teamcolorcodes.com (Penticton: Vees blue / red)
WHL_CLUBS = {
    "brandon": ("#E6BF2E", "#000000", 49.849, -99.967), "calgary": ("#E13A3E", "#000000", 51.037, -114.052),
    "edmonton": ("#00549B", "#FFC42A", 53.547, -113.498), "everett": ("#045B41", "#BC8448", 47.978, -122.202),
    "kamloops": ("#D0471D", "#032C62", 50.674, -120.334), "kelowna": ("#008394", "#E13A3E", 49.886, -119.497),
    "lethbridge": ("#CC1D30", "#121942", 49.690, -112.840), "medicine hat": ("#E56B1F", "#000000", 50.040, -110.710),
    "moose jaw": ("#CC0000", "#000000", 50.393, -105.535), "penticton": ("#00539B", "#C8102E", 49.490, -119.590),
    "portland": ("#CC202E", "#000000", 45.531, -122.668), "prince albert": ("#0D843E", "#DAC796", 53.200, -105.750),
    "prince george": ("#D0202E", "#D0A176", 53.910, -122.750), "red deer": ("#900028", "#A7B1B7", 52.270, -113.810),
    "regina": ("#1A3668", "#EF3342", 50.440, -104.620), "saskatoon": ("#FFCD36", "#262266", 52.180, -106.700),
    "seattle": ("#003876", "#008357", 47.380, -122.230), "spokane": ("#C41230", "#013A81", 47.666, -117.420),
    "swift current": ("#026937", "#001F5B", 50.290, -107.800), "tri-city": ("#0D1D41", "#AC1A2D", 46.210, -119.170),
    "vancouver": ("#97002E", "#B5B7B9", 49.100, -122.650), "victoria": ("#013974", "#A4ADB4", 48.430, -123.360),
    "wenatchee": ("#002F87", "#026FA8", 47.420, -120.310),
}
WHL_CODE_HINT = {"BDN": "brandon", "BRN": "brandon", "CGY": "calgary", "CAL": "calgary", "EDM": "edmonton",
                 "EVT": "everett", "EVE": "everett", "KAM": "kamloops", "KEL": "kelowna", "LET": "lethbridge",
                 "LETH": "lethbridge", "MH": "medicine hat", "MED": "medicine hat", "MJ": "moose jaw",
                 "PEN": "penticton", "POR": "portland", "PA": "prince albert", "PAR": "prince albert",
                 "PG": "prince george", "RD": "red deer", "REG": "regina", "SAS": "saskatoon", "SSK": "saskatoon",
                 "SEA": "seattle", "SPO": "spokane", "SC": "swift current", "TC": "tri-city", "TRI": "tri-city",
                 "VAN": "vancouver", "VIC": "victoria", "WEN": "wenatchee"}


def club_key(team):
    try:
        name = str(teams_info().get(team, {}).get("name", "")).lower()
    except Exception:
        name = ""
    for k in WHL_CLUBS:
        if k in name or k.replace("-", " ") in name:
            return k
    return WHL_CODE_HINT.get(str(team).upper())


def apply_team_colors(teams):
    for i, t in enumerate(teams):
        k = club_key(t)
        TEAM_COLORS[t] = WHL_CLUBS[k][0] if k else PALETTE[i % len(PALETTE)]
        TEAM_COLORS2[t] = WHL_CLUBS[k][1] if k else "#8A8D8F"


TEAM_COLORS2 = {}


def team_colors(home, away):
    """Home/away colours; if they look alike, the away team switches to its second colour."""
    def rgb(h):
        h = h.lstrip("#")
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)])
    ch, ca = TEAM_COLORS.get(home, "#1f77b4"), TEAM_COLORS.get(away, "#d62728")
    if np.abs(rgb(ch) - rgb(ca)).sum() < 120:
        alt = TEAM_COLORS2.get(away, "#8A8D8F")
        if np.abs(rgb(ch) - rgb(alt)).sum() < 120 or rgb(alt).sum() < 90:
            alt = "#8A8D8F" if np.abs(rgb(ch) - rgb("#8A8D8F")).sum() > 150 else "#E07A00"
        ca = alt
    return ch, ca


def arena_km(home_a, home_b):
    """Road distance (km, roughly 1.25 x straight line) between two teams' arenas."""
    ka, kb = club_key(home_a), club_key(home_b)
    if not ka or not kb or ka == kb:
        return 0.0
    la1, lo1 = map(math.radians, WHL_CLUBS[ka][2:4])
    la2, lo2 = map(math.radians, WHL_CLUBS[kb][2:4])
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 1.25 * 2 * 6371 * math.asin(math.sqrt(h))


FATIGUE_LEVELS = {"Fresh": (1.02, 0.98), "Normal": (1.0, 1.0), "Tired": (0.97, 1.03), "Exhausted": (0.94, 1.06)}


def _team_dates(team):
    try:
        return [(g["date"], g["home"]) for g in season_schedule(season_id()) if team in (g["home"], g["away"])]
    except Exception:
        return []


def fatigue_info(team, game_date, game_home=None):
    """How tired a team is before a game: back-to-backs, three-in-three weekends and bus trips.
    Returns factors for its own shot rate (f_for) and for what it allows (f_ag)."""
    out = dict(b2b=False, three=False, km=0.0, rest=None, road=0, f_for=1.0, f_ag=1.0, text="", level="Normal")
    try:
        d0 = dt.date.fromisoformat(str(game_date)[:10])
    except Exception:
        return out
    sched = _team_dates(team)
    prev = [(dt.date.fromisoformat(d), h) for d, h in sched if d < str(d0)]
    if not prev:
        return out
    last_d, last_home = prev[-1]
    rest = (d0 - last_d).days
    out["rest"] = rest
    out["b2b"] = rest == 1
    out["three"] = rest == 1 and sum(1 for d, _ in prev[-3:] if (d0 - d).days <= 2) >= 2
    if game_home is None:
        game_home = next((h for d, h in sched if d == str(d0)), team)
    out["km"] = arena_km(last_home, game_home) if rest <= 2 else 0.0
    road = 0
    for d, h in reversed(prev):
        if h == team:
            break
        road += 1
    out["road"] = road + (game_home != team)
    f_for = f_ag = 1.0
    bits = []
    if out["b2b"]:
        f_for, f_ag = B2B_FOR, B2B_AG
        bits.append("3rd game in 3 nights" if out["three"] else "2nd night of a back-to-back")
        if out["three"]:
            f_for, f_ag = f_for * 0.98, f_ag * 1.02
    if out["km"] >= 1200:
        f_for, f_ag = f_for * 0.975, f_ag * 1.025
    elif out["km"] >= 600:
        f_for, f_ag = f_for * 0.985, f_ag * 1.015
    if out["km"] >= 600:
        bits.append(f"{out['km']:,.0f} km trip since the last game")
    if out["road"] >= 4:
        f_for, f_ag = f_for * 0.99, f_ag * 1.01
        bits.append(f"game {out['road']} of a road trip")
    if rest >= 3 and not bits:
        bits.append(f"rested ({rest} days off)")
    out.update(f_for=f_for, f_ag=f_ag, text=" · ".join(bits))
    out["level"] = ("Exhausted" if f_for < 0.95 else "Tired" if f_for < 0.995 else "Fresh" if rest >= 3 else "Normal")
    return out


def fatigue_mult(team, game_date, game_home=None):
    fi = fatigue_info(team, game_date, game_home)
    return fi["f_for"], fi["f_ag"]


def team_lines(prof):
    """No public source publishes WHL line combinations, so lineups come from recent game sheets."""
    return None


def guess_goalie(prof, game_date=None):
    """(goalie id, reason): who started recently, with rest and back-to-backs taken into account."""
    gk = {g["id"]: g for g in prof["goalies"]}
    starts = [(rg["date"], s) for rg, s in zip(prof["recent_games"], prof["starters"]) if s in gk]
    if not starts:
        if not prof["goalies"]:
            return None, ""
        g = max(prof["goalies"], key=lambda x: (x["gs"], x["gp"]))
        return g["id"], "most starts this season"
    last10 = starts[-10:]
    c = Counter(s for _, s in last10)
    topn = max(c.values())
    top = next(s for _, s in reversed(last10) if c[s] == topn)
    others = [s for s, _ in c.most_common() if s != top]
    if not others:
        others = [g["id"] for g in sorted(prof["goalies"], key=lambda x: -x["gs"]) if g["id"] != top and g["gs"] > 0]
    try:
        d0 = dt.date.fromisoformat(str(game_date)[:10]) if game_date else None
    except Exception:
        d0 = None
    nm = lambda gid: gk.get(gid, {}).get("name", "?")
    if d0 and others:
        recent = [(dt.date.fromisoformat(str(d)[:10]), s) for d, s in starts[-2:]]
        last_d, last_s = recent[-1]
        if (d0 - last_d).days == 1 and last_s == top:
            two_in_a_row = len(recent) == 2 and (last_d - recent[0][0]).days == 1 and recent[0][1] == top
            if two_in_a_row or topn / len(last10) < 0.7:
                return others[0], (f"{nm(top)} started " + ("the last two nights" if two_in_a_row else "last night")
                                   + ", so the backup usually gets this one")
    rest = ""
    if d0:
        last_top = max((dt.date.fromisoformat(str(d)[:10]) for d, s in starts if s == top), default=None)
        if last_top:
            days = (d0 - last_top).days
            rest = f" · {days} day{'s' if days != 1 else ''} since his last start" if days > 0 else ""
    return top, f"started {topn} of the last {len(last10)}{rest}"


# --------------------------------------------------------------------------
# Jersey numbers (for the animation) and player names from game sheets
# --------------------------------------------------------------------------
def jersey_numbers():
    out = {}
    try:
        sk, gk = league_players()
    except Exception:
        return out
    for lst in list(sk.values()) + list(gk.values()):
        for p in lst:
            if p.get("num"):
                out[str(p["id"])] = str(p["num"])
    return out


# --------------------------------------------------------------------------
# Beat the Bot pick'em, model-picks record, title-odds history
# --------------------------------------------------------------------------
BOT = "🤖 The Bot"
PICK_COLS = ["pick_id", "date", "game_id", "start_utc", "away", "home", "player", "pick", "score_away", "score_home",
             "bot_p", "created_at", "status", "final_away", "final_home", "points", "note"]
PICK_TEXT = ["pick_id", "date", "start_utc", "away", "home", "player", "pick", "created_at", "status", "note"]
ODDSH_COLS = ["date", "team", "proj_pts", "playoffs", "conf_final", "final", "champ"]
ODDSH_TEXT = ["date", "team"]
STORES["pickem"] = dict(cols=PICK_COLS, text=PICK_TEXT, file="whl_pickem.csv", secret="WHL_PICKEM_PATH")
STORES["odds_history"] = dict(cols=ODDSH_COLS, text=ODDSH_TEXT, file="whl_title_odds.csv", secret="WHL_ODDS_HISTORY_PATH")


def store_update(store, fn, message, tries=4):
    """Read the newest copy, change it with fn(df) -> (df, changed), save; retries if someone else saved first
    (friends can play at the same time)."""
    import time
    for k in range(tries):
        load_store.clear()
        df = load_store(store).copy()
        df, changed = fn(df)
        if not changed:
            return 0
        try:
            save_store(store, df, message)
            return changed
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", 0)
            if code in (409, 422) and k < tries - 1:
                time.sleep(1.0 + k)
                continue
            raise
    return 0


def pick_points(r):
    """Points for one graded pick: 10 for the winner, +5 for an upset the bot didn't see coming,
    +15 for the exact score, +5 for a close score (each team within one goal)."""
    fa, fh = int(r["final_away"]), int(r["final_home"])
    winner = r["home"] if fh > fa else r["away"]
    right = r["pick"] == winner
    pts, notes = 0, []
    if right:
        pts += 10
        notes.append("✅ winner")
        try:
            if float(r["bot_p"]) < 0.4:
                pts += 5
                notes.append("🦄 upset +5")
        except (TypeError, ValueError):
            pass
    else:
        notes.append("❌ winner")
    try:
        sa, sh = int(r["score_away"]), int(r["score_home"])
        if (sa, sh) == (fa, fh):
            pts += 15
            notes.append("🎯 exact score +15")
        elif abs(sa - fa) <= 1 and abs(sh - fh) <= 1:
            pts += 5
            notes.append("👌 close score +5")
    except (TypeError, ValueError):
        pass
    return ("Won" if right else "Lost"), pts, " · ".join(notes)


def grade_pickem():
    df = load_store("pickem")
    todo = df[(df["status"] == "Open") & df["start_utc"].apply(_started)]
    if todo.empty:
        return df, 0
    finals = {}
    for d in todo["date"].unique():
        finals.update(final_scores(str(d)))

    def fn(df):
        changed = 0
        for i, r in df.iterrows():
            if r["status"] != "Open":
                continue
            sc = finals.get(int(r["game_id"]))
            if not sc or sc[0] is None:
                continue
            df.loc[i, "final_away"], df.loc[i, "final_home"] = int(sc[0]), int(sc[1])
            status, pts, note = pick_points(df.loc[i])
            df.loc[i, "status"], df.loc[i, "points"], df.loc[i, "note"] = status, pts, note
            changed += 1
        return df, changed

    n = store_update("pickem", fn, "Grade pick'em")
    return load_store("pickem"), n


def save_picks(rows, message):
    """Add or change picks (one per player per game). Games that have started are locked."""
    def fn(df):
        changed = 0
        for r in rows:
            if _started(r["start_utc"]):
                continue
            hit = df.index[(df["player"] == r["player"]) & (pd.to_numeric(df["game_id"], errors="coerce") == r["game_id"])]
            if len(hit):
                old = df.loc[hit[0]]
                same = (old["pick"] == r["pick"] and str(old["score_away"]) == str(r.get("score_away"))
                        and str(old["score_home"]) == str(r.get("score_home")))
                if same or _started(old["start_utc"]):
                    continue
                df = df.drop(hit)
            df = pd.concat([df, pd.DataFrame([r])[PICK_COLS]], ignore_index=True)
            changed += 1
        return df.sort_values(["date", "start_utc", "game_id", "player"]).reset_index(drop=True), changed
    return store_update("pickem", fn, message)


def pick_row(player, game, pick, score_away=None, score_home=None, bot_p=None):
    return dict(pick_id=f"{game['id']}-{player}", date=game["date"], game_id=int(game["id"]),
                start_utc=game.get("start", ""), away=game["away"], home=game["home"], player=player, pick=pick,
                score_away=score_away, score_home=score_home, bot_p=None if bot_p is None else round(float(bot_p), 4),
                created_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), status="Open",
                final_away=None, final_home=None, points=None, note="")


def bot_rows(day_str, preds):
    """The model's own picks (winner + most likely score) for games that haven't started."""
    rows = []
    for gid, p in preds.items():
        if "p_home" not in p or _started(p.get("start", "")):
            continue
        ph = float(p["p_home"])
        pick = p["home"] if ph >= 0.5 else p["away"]
        try:
            sa, sh = (int(x) for x in str(p.get("likely_ah", "")).split("-"))
        except ValueError:
            sa = sh = None
        if sa is not None and (sa == sh or (sh > sa) != (pick == p["home"])):
            sa, sh = (min(sa, sh), max(sa, sh) + (sa == sh)) if pick == p["home"] else (max(sa, sh) + (sa == sh), min(sa, sh))
        rows.append(pick_row(BOT, dict(id=gid, date=day_str, start=p.get("start", ""), away=p["away"], home=p["home"]),
                             pick, sa, sh, ph if pick == p["home"] else 1 - ph))
    return rows


def pickem_table(df):
    """Season leaderboard with streaks and badges."""
    g = df[df["status"].isin(["Won", "Lost"])].copy()
    if g.empty:
        return pd.DataFrame()
    g["points"] = pd.to_numeric(g["points"], errors="coerce").fillna(0)
    g = g.sort_values(["date", "start_utc", "game_id"])
    nightly = g.groupby(["date", "player"])["points"].sum().unstack(fill_value=0)
    rows = []
    for player, grp in g.groupby("player"):
        won = (grp["status"] == "Won").to_numpy()
        cur = best = run = 0
        for w in won:
            run = run + 1 if w else 0
            best = max(best, run)
        cur = run
        exact = int(grp["note"].fillna("").str.contains("exact").sum())
        upsets = int(grp["note"].fillna("").str.contains("upset").sum())
        nights = grp["date"].nunique()
        perfect = int(sum(1 for _, n in grp.groupby("date") if len(n) >= 3 and (n["status"] == "Won").all()))
        beat = 0
        if player != BOT and BOT in nightly.columns:
            both = nightly[[player, BOT]]
            played = grp["date"].unique()
            beat = int((both.loc[played, player] > both.loc[played, BOT]).sum())
        badges = "".join(b for b, ok in (("🔥", best >= 5), ("🎯", exact >= 1), ("🦄", upsets >= 3), ("💯", perfect >= 1),
                                         ("🤖💥", beat >= 1), ("📅", nights >= 10)) if ok)
        rows.append({"Player": player, "Points": int(grp["points"].sum()), "Picks": len(grp),
                     "Right": int(won.sum()), "Win %": 100 * won.mean(), "Exact scores": exact, "Upsets": upsets,
                     "Streak": cur, "Best streak": best, "Nights beat the Bot": beat if player != BOT else np.nan,
                     "Badges": badges})
    out = pd.DataFrame(rows).sort_values(["Points", "Win %"], ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", range(1, len(out) + 1))
    return out


BADGES = [("🔥", "Hot streak", "5 right in a row"), ("🎯", "Sniper", "an exact final score"),
          ("🦄", "Upset artist", "3 upsets the bot didn't see coming"), ("💯", "Perfect night", "every winner right on a 3+ game night"),
          ("🤖💥", "Bot beater", "more points than the bot on a night"), ("📅", "Regular", "picks on 10 different nights")]


def save_title_odds(tbl):
    """Keep one snapshot a day of every team's playoff and championship odds (for the race chart)."""
    today = local_today().isoformat() if "local_today" in globals() else dt.date.today().isoformat()
    snap = pd.DataFrame(dict(date=today, team=tbl["Team"], proj_pts=tbl["Proj PTS"].round(1),
                             playoffs=tbl["Playoffs %"].round(2), conf_final=tbl["Conf final %"].round(2),
                             final=tbl["Final %"].round(2), champ=tbl["Cup %"].round(2)))

    def fn(df):
        df = pd.concat([df[df["date"] != today], snap[ODDSH_COLS]], ignore_index=True)
        return df.sort_values(["date", "team"]).reset_index(drop=True), len(snap)
    return store_update("odds_history", fn, f"Title odds {today}")


# --------------------------------------------------------------------------
# Scoring race: chance each top scorer wins the WHL scoring title
# --------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def scoring_race(stat="Points", top=15, n=20000, seed=3):
    sk, _ = league_players()
    players = [dict(p, team=t) for t, lst in sk.items() for p in lst]
    if not players:
        return pd.DataFrame()
    val = (lambda p: p["g"] + p["a"]) if stat == "Points" else (lambda p: p["g"])
    players.sort(key=lambda p: (-val(p), p["gp"]))
    pool = players[:max(top, 25)]
    try:
        left = Counter()
        for g in season_schedule(season_id()):
            if g["state"] != "FINAL":
                left[g["home"]] += 1
                left[g["away"]] += 1
    except Exception:
        left = Counter()
    cur = np.array([val(p) for p in pool], float)
    gp = np.array([max(p["gp"], 1) for p in pool], float)
    rate_pop = float((cur / gp).mean())
    rate = (cur + rate_pop * 8) / (gp + 8)                  # shrink hot starts toward the group's pace
    rem = np.array([left.get(p["team"], 0) for p in pool])
    rng = np.random.default_rng(seed)
    games = rng.binomial(rem[None, :].repeat(n, 0), 0.92)  # some games missed (injuries, World Juniors, etc.)
    final = cur[None, :] + rng.poisson(rate[None, :] * games)
    best = final.max(1, keepdims=True)
    lead = final == best
    share = lead / lead.sum(1, keepdims=True)
    top3 = (final >= np.sort(final, 1)[:, -3][:, None])
    df = pd.DataFrame({"id": [p["id"] for p in pool], "Player": [p["name"] for p in pool],
                       "Team": [p["team"] for p in pool], "Pos": [p["pos"] for p in pool], "GP": gp.astype(int),
                       "G": [p["g"] for p in pool], "A": [p["a"] for p in pool], stat: cur.astype(int),
                       "Per game": cur / gp, "Games left": rem, "Projected": final.mean(0),
                       "Title %": 100 * share.mean(0), "Top 3 %": 100 * top3.mean(0)})
    return df.sort_values(["Title %", stat], ascending=False).head(top).reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def points_by_date(ids, stat="Points"):
    """{player id: [(date, running total)]} from each player's game log."""
    out = {}
    key = "points" if stat == "Points" else "goals"
    for pid in ids:
        rows = sorted(_player_log(pid), key=lambda r: str(r.get("date_played", "")))
        tot, seq = 0, []
        for r in rows:
            v = r.get(key)
            if v in (None, "") and key == "points":
                v = _num(r.get("goals")) + _num(r.get("assists"))
            tot += int(_num(v))
            seq.append((str(r.get("date_played", ""))[:10], tot))
        out[pid] = seq
    return out


# --------------------------------------------------------------------------
# Rivalries: every meeting between two teams this season and last
# --------------------------------------------------------------------------
def meetings(a, b):
    rows = []
    for label, sid in (("This season", season_id()), ("Last season", prev_season_id())):
        if not sid:
            continue
        try:
            for g in season_schedule(sid):
                if {g["home"], g["away"]} == {a, b}:
                    rows.append(dict(g, season=label))
        except Exception:
            pass
    return sorted(rows, key=lambda g: (g["date"], g["id"]))


# ==========================================================================
# APP  (multipage layout that works on phones and computers)
# ==========================================================================
ss = st.session_state
PCFG = dict(displayModeBar=False, responsive=True)

BASE_CSS = """<style>
.block-container { padding-top: 3.6rem; padding-bottom: 4rem; max-width: 1400px; }
@media (max-width: 640px) {
  .block-container { padding-left: .65rem; padding-right: .65rem; padding-top: 3.2rem; }
  h1 { font-size: 1.45rem !important; } h2 { font-size: 1.25rem !important; } h3 { font-size: 1.08rem !important; }
  .stTabs [data-baseweb="tab"] { padding-left: .45rem; padding-right: .45rem; font-size: .86rem; }
  [data-testid="stMetricValue"] { font-size: 1.25rem; }
}
.kgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 9px; margin: 4px 0 14px; }
.kpi { background: rgba(128,128,128,.09); border-radius: 12px; padding: 9px 12px; min-width: 0; }
.kpi .l { font-size: .76rem; opacity: .72; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kpi .v { font-size: 1.3rem; font-weight: 800; line-height: 1.25; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kpi .s { font-size: .74rem; opacity: .66; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kpi.up .v { color: #00b85c; } .kpi.down .v { color: #e5484d; }
.gc .tm { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin: 3px 0; font-weight: 700; font-size: 1.02rem; }
.gc .tm span.n { display: flex; align-items: center; gap: 8px; }
.gc .stripe { width: 4px; height: 26px; border-radius: 3px; display: inline-block; }
.gc img { width: 30px; height: 30px; }
.gc .num { font-weight: 800; font-variant-numeric: tabular-nums; }
.gc .num.fav { color: #00b85c; }
.gc .meta { font-size: .78rem; opacity: .72; margin-top: 4px; }
.gc .bar { height: 7px; border-radius: 4px; overflow: hidden; display: flex; margin-top: 6px; background: rgba(128,128,128,.2); }
.badge { display: inline-block; padding: 1px 8px; border-radius: 8px; font-size: .72rem; font-weight: 800; letter-spacing: .3px; }
.b-live { background: #e53935; color: #fff; } .b-final { background: rgba(128,128,128,.25); } .b-fut { background: rgba(46,134,222,.18); color: #2e86de; }
.b-ok { background: rgba(39,174,96,.18); color: #1e8449; } .b-bad { background: rgba(192,57,43,.15); color: #c0392b; }
.pc { display: flex; gap: 14px; align-items: center; margin-bottom: 6px; }
.pc img { width: 76px; height: 76px; border-radius: 50%; background: rgba(128,128,128,.15); object-fit: cover; }
.pc .nm { font-size: 1.25rem; font-weight: 800; line-height: 1.2; } .pc .sub { font-size: .85rem; opacity: .75; }
.hero { display: flex; align-items: center; justify-content: center; gap: 14px; flex-wrap: wrap; margin: 2px 0 6px; }
.hero img { width: 46px; height: 46px; } .hero .at { opacity: .6; font-weight: 700; }
.hero .ab { font-size: 1.5rem; font-weight: 900; }
.small { font-size: .82rem; opacity: .8; }
.lines { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 6px 14px; font-size: .86rem; }
.lines .row { display: flex; gap: 8px; align-items: baseline; }
.lines .tag { font-weight: 800; font-size: .72rem; min-width: 30px; opacity: .7; }
.slip { border-radius: 14px; padding: 10px 14px; margin: 6px 0; background: rgba(128,128,128,.08); }
.slip .leg { display: flex; justify-content: space-between; gap: 10px; padding: 3px 0; font-size: .92rem; }
.slip .leg span:last-child { font-variant-numeric: tabular-nums; opacity: .85; }
</style>"""

DARK_CSS = """<style>
.stApp { background: radial-gradient(1100px 520px at 12% -8%, #17244a 0%, rgba(11,18,32,0) 60%),
                     radial-gradient(900px 480px at 100% 0%, #10302a 0%, rgba(11,18,32,0) 55%), #0b1220; }
[data-testid="stHeader"] { background: rgba(11,18,32,.82); backdrop-filter: blur(10px); }
h1, h2, h3, h4 { letter-spacing: .2px; }
div[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stVerticalBlock"][class*="border"] {
  background: linear-gradient(180deg, #131d34, #0f182c); border-color: #24304d !important;
  box-shadow: 0 8px 22px rgba(0,0,0,.35); }
.kpi { background: linear-gradient(180deg, #17223f, #111a30); border: 1px solid #22304d; box-shadow: 0 4px 14px rgba(0,0,0,.25); }
.kpi .v { color: #ffffff; } .kpi.up .v { color: #00e676; } .kpi.down .v { color: #ff5c7a; }
.gc .num.fav { color: #00e676; }
.b-fut { background: rgba(79,195,255,.14); color: #4fc3ff; } .b-final { background: rgba(255,255,255,.1); color: #c8d3ea; }
.b-ok { background: rgba(0,230,118,.14); color: #00e676; } .b-bad { background: rgba(255,92,122,.14); color: #ff5c7a; }
.slip { background: linear-gradient(180deg, #17223f, #111a30); border: 1px solid #22304d; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [aria-selected="true"] { color: #00e676 !important; }
div[data-testid="stPopover"] button, .stButton button { font-weight: 700; }
</style>"""


def theme_is_dark():
    try:
        return (st.context.theme.type or "dark") == "dark"
    except Exception:
        return True


LOGO_DARK = theme_is_dark()
st.markdown(BASE_CSS + (DARK_CSS if LOGO_DARK else ""), unsafe_allow_html=True)


def esc(x):
    return html_lib.escape(str(x))


def kpis(items):
    """Responsive stat tiles: 2 per row on a phone, up to 6 on a computer.
    items = (label, value, sub) or (label, value, sub, 'up'|'down')."""
    cells = "".join(f'<div class="kpi {it[3] if len(it) > 3 else ""}"><div class="l">{esc(it[0])}</div>'
                    f'<div class="v">{esc(it[1])}</div>' + (f'<div class="s">{esc(it[2])}</div>' if it[2] else "") + "</div>"
                    for it in items)
    st.markdown(f'<div class="kgrid">{cells}</div>', unsafe_allow_html=True)


def chart(fig, key, height=None):
    if height:
        fig.update_layout(height=height)
    fig.update_layout(margin=dict(l=8, r=8, t=fig.layout.margin.t if fig.layout.margin.t is not None else 30, b=8))
    st.plotly_chart(fig, width="stretch", config=PCFG, key=key)


def local_tz():
    if os.environ.get("WHL_TZ"):
        return os.environ["WHL_TZ"]
    try:
        tz = st.context.timezone
        if tz:
            return tz
    except Exception:
        pass
    return "America/Los_Angeles"


def local_today():
    """Today's date where the user is (the server clock is UTC, which is 'tomorrow' on North American evenings)."""
    try:
        return pd.Timestamp.now(tz=local_tz()).date()
    except Exception:
        return dt.date.today()


def local_time(utc):
    try:
        t = pd.Timestamp(utc).tz_convert(local_tz())
        return t.strftime("%I:%M %p").lstrip("0") + " " + t.strftime("%Z")
    except Exception:
        return ""


def fmt_odds(v):
    try:
        return f"{int(v):+d}"
    except Exception:
        return "–"


def money(v, sign=True):
    v = float(v)
    return ("+" if sign and v > 0 else "-" if v < 0 else "") + f"${abs(v):,.2f}"


def pct1(p):
    return "–" if p is None or (isinstance(p, float) and np.isnan(p)) else f"{100 * p:.1f}%"


try:
    standings = load_standings()
    TEAMS = standings["team"].tolist()
except Exception as e:
    st.error(f"Couldn't reach the WHL data feed ({e}). Check your internet connection and refresh.")
    st.stop()
if not TEAMS:
    st.error("The WHL feed came back with no teams. Try again in a minute.")
    st.stop()
NAMES = dict(zip(standings["team"], standings["name"]))
STN = standings.set_index("team")
apply_team_colors(TEAMS)
_divs = [d for d in standings["div"].dropna().unique() if d]
if _divs:
    _cd = {}
    for d in _divs:
        _cd.setdefault(conference_of(d), []).append(d)
    CONF_DIVS = {c: tuple(sorted(v)) for c, v in _cd.items()}
    CONF_OF = {d: c for c, ds in CONF_DIVS.items() for d in ds}


def _label(t):
    return f"{t} – {NAMES.get(t, t)}"


# Settings that survive switching pages
for k, v in (("n_sims", 5000), ("n_recent", 15), ("playoff", False), ("bankroll", 1000.0), ("use_lines", False),
             ("home", "SEA" if "SEA" in TEAMS else TEAMS[0]), ("away", "POR" if "POR" in TEAMS else TEAMS[1]),
             ("day", local_today())):
    ss.setdefault(k, v)
if ss["home"] not in TEAMS:
    ss["home"] = TEAMS[0]
if ss["away"] not in TEAMS:
    ss["away"] = TEAMS[1]


def _keep(k):
    ss[k] = ss["_w_" + k]


def settings_popover():
    with st.popover("⚙️ Settings", width="stretch"):
        st.select_slider("Simulations per game", [1000, 2000, 5000, 10000, 20000], value=ss["n_sims"],
                         key="_w_n_sims", on_change=_keep, args=("n_sims",),
                         help="More simulations = steadier numbers, but slower.")
        st.slider("Recent games that shape each team's style", 5, 30, value=ss["n_recent"], key="_w_n_recent",
                  on_change=_keep, args=("n_recent",),
                  help="Newer games count more (a game 10 games ago counts half as much as the last one).")
        st.toggle("Playoff rules (sudden-death 5-on-5 OT, no shootout)", value=ss["playoff"], key="_w_playoff",
                  on_change=_keep, args=("playoff",))
        if not theme_is_dark():
            st.caption("🌙 For the dark look, choose Dark under ⋮ → Settings, or add the .streamlit/config.toml file.")
        st.caption("Data: WHL public stats feed · refreshes every 15–30 min (live scores every 45 s).")


def open_game(home, away, game=None):
    ss["home"], ss["away"] = home, away
    ss["gc_game"] = dict(home=home, away=away, **game) if game else None
    for k in ("_w_home", "_w_away"):
        ss.pop(k, None)


def odds_key_input(where):
    st.caption("Sportsbook odds for WHL games aren't available from an odds feed, so compare the fair odds "
               "with any line you find yourself. You can still log every WHL bet you make here and it's graded "
               "automatically after the game.")
    return ""


def team_hero(away, home, sub=""):
    st.markdown(f'<div class="hero"><img src="{logo_url(away, LOGO_DARK)}"><span class="ab">{away}</span>'
                f'<span class="at">@</span><span class="ab">{home}</span><img src="{logo_url(home, LOGO_DARK)}"></div>'
                + (f'<div class="small" style="text-align:center">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


SHORT_ICON = {"Confirmed": "✅", "Likely": "🟡", "Projected": "⚪", "Recent": "🔁", "Manual": "✏️"}


# ---------------- simulating games ----------------
def matchup_inputs(home, away):
    """Profiles, next game, confirmed goalies and the lineup in use for a matchup."""
    n_recent = ss["n_recent"]
    hp, ap = build_profile(home, n_recent), build_profile(away, n_recent)
    pinned = ss.get("gc_game")
    if pinned and pinned.get("home") == home and pinned.get("away") == away:
        ng = {k: pinned[k] for k in ("id", "date", "start", "playoff")}     # the exact game opened from 📅 Games
    else:
        ng = next_game(home, away)
    gs = starting_goalies(home, away, ng["date"] if ng else None, hp, ap)
    key = f"lu_{away}_{home}_{int(bool(ss.get('use_lines', True)))}"
    ov = ss.get(key, {})
    ul = ss.get("use_lines", True)
    return dict(home=home, away=away, hp=hp, ap=ap, ng=ng, gs=gs, key=key,
                hg=ov.get("hg", gs["home"][0]), ag=ov.get("ag", gs["away"][0]),
                ho=ov.get("ho", default_lineup(hp, ul)[1]), ao=ov.get("ao", default_lineup(ap, ul)[1]))


def sim_for(mi, n=None):
    playoff = bool(ss["playoff"] or (mi["ng"] and mi["ng"].get("playoff")))
    return run_matchup(mi["home"], mi["away"], mi["hg"], mi["ag"], mi["ho"], mi["ao"], n or ss["n_sims"], playoff,
                       ss["n_recent"], None, mi["ng"]["date"] if mi["ng"] else None, ss["use_lines"])


def sim_game(g, day_str, n):
    """Simulate one scheduled game with confirmed goalies and the usual lineups."""
    h, a = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
    hpp, app = build_profile(h, ss["n_recent"]), build_profile(a, ss["n_recent"])
    gs = starting_goalies(h, a, day_str, hpp, app)
    ul = ss.get("use_lines", True)
    res = run_matchup(h, a, gs["home"][0], gs["away"][0], default_lineup(hpp, ul)[1], default_lineup(app, ul)[1], n,
                      g.get("gameType") == 3, ss["n_recent"], None, day_str, ul)
    return res, gs, hpp, app


def goalie_tag(prof, gid, status):
    return f"{goalie_name(prof, gid)} {SHORT_ICON.get(status, '')}"


def simulate_slate(day_str, games, n, save, progress=None):
    """Simulate every game on a date; keep the results for this session and optionally save them."""
    out = ss.setdefault("slate", {}).setdefault(day_str, {})
    plays = ss.setdefault("slate_players", {}).setdefault(day_str, {})
    records, prec, evts = [], [], []
    key = odds_api_key() if save else ""
    if key:
        try:
            evts = odds_feed(key)[0]
        except Exception:
            evts = []
    for k, g in enumerate(games):
        h, a = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
        try:
            res, gs, hpp, app = sim_game(g, day_str, n)
            game = dict(id=g["id"], date=day_str, start=g.get("startTimeUTC", ""))
            ts = res["top_scores"][0][0]
            out[g["id"]] = dict(home=h, away=a, p_home=float(res["home_win"]), gh=float(res["goals_mean"][0]),
                                start=g.get("startTimeUTC", ""), likely_ah=f"{ts[1]}-{ts[0]}",
                                fat=fatigue_text(res.get("fatigue") or {}, h, a),
                                ga=float(res["goals_mean"][1]), likely=f"{a} {ts[1]}–{ts[0]} {h}",
                                ot=float(res["ot"] + res["so"]), total=float(res["total"].mean()),
                                hg=goalie_tag(hpp, gs["home"][0], gs["home"][1]),
                                ag=goalie_tag(app, gs["away"][0], gs["away"][1]), b2b=res["b2b"])
            prow = player_records(res, game, top=18)
            plays[g["id"]] = prow
            if save:
                gnames = {sd: (goalie_name(hpp if sd == "home" else app, gs[sd][0]), gs[sd][1]) for sd in ("home", "away")}
                records.append(make_record(res, game, gnames, ss["n_recent"], "slate", find_event(evts, h, a)))
                prec += player_records(res, game)
        except Exception as e:
            out[g["id"]] = dict(home=h, away=a, error=str(e))
        if progress is not None:
            progress.progress((k + 1) / len(games), text=f"Simulated {a} @ {h}")
    saved = saved_p = 0
    if save:
        saved = upsert_predictions(records, f"Save {len(records)} WHL predictions for {day_str}") if records else 0
        saved_p = upsert_players(prec, f"Save player predictions for {day_str}") if prec else 0
    return saved, saved_p


@st.cache_data(ttl=1800, show_spinner=False)
def day_games_safe(day_str):
    try:
        return day_games(day_str)
    except Exception:
        return []


def todays_matchups(day_str):
    """[(label, game dict)] for the date, used by pickers on several pages."""
    return [(f"{g['awayTeam']['abbrev']} @ {g['homeTeam']['abbrev']}" + (f" · {local_time(g.get('startTimeUTC'))}"
             if g.get("startTimeUTC") else ""), g) for g in day_games_safe(day_str)]


def day_games(day_str):
    return scoreboard(day_str)


# ==========================================================================
# PAGE: Today
# ==========================================================================
def game_card(g, live, sim, day_str):
    h, a = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
    ch, ca = team_colors(h, a)
    state = (live or {}).get("gameState", g.get("gameState", "FUT"))
    hs = (live or {}).get("homeTeam", {}).get("score")
    as_ = (live or {}).get("awayTeam", {}).get("score")
    p_home, p_lbl = None, ""
    if state in ("LIVE", "CRIT"):
        elapsed, clk = game_clock(live)
        badge = f'<span class="badge b-live">LIVE · {esc(clk)}</span>'
        try:
            p_home = live_prob(h, a, elapsed, int(hs or 0), int(as_ or 0), ss["n_recent"], g.get("gameType") == 3)
            p_lbl = "Live win chance"
        except Exception:
            p_home = None
    elif state in ("OFF", "FINAL"):
        lp = ((live or {}).get("gameOutcome") or {}).get("lastPeriodType", "REG")
        badge = f'<span class="badge b-final">FINAL{"" if lp == "REG" else " / " + esc(lp)}</span>'
        if sim and "p_home" in sim and hs is not None:
            right = (sim["p_home"] >= 0.5) == (hs > as_)
            badge += f' <span class="badge {"b-ok" if right else "b-bad"}">model {"✓" if right else "✗"}</span>'
    else:
        badge = f'<span class="badge b-fut">{esc(local_time(g.get("startTimeUTC")) or "Scheduled")}</span>'
    if p_home is None and sim and "p_home" in sim:
        p_home, p_lbl = sim["p_home"], "Model win chance"

    def num(side):
        if state in ("LIVE", "CRIT", "OFF", "FINAL") and hs is not None:
            return str(hs if side == "h" else as_)
        if p_home is not None:
            return f"{100 * (p_home if side == 'h' else 1 - p_home):.0f}%"
        return ""

    def fav(side):
        if state in ("LIVE", "CRIT", "OFF", "FINAL") and hs is not None:
            return (hs > as_) if side == "h" else (as_ > hs)
        return p_home is not None and ((p_home >= 0.5) if side == "h" else (p_home < 0.5))

    rows = "".join(
        f'<div class="tm"><span class="n"><span class="stripe" style="background:{col}"></span>'
        f'<img src="{logo_url(t, LOGO_DARK)}" alt="">{t}'
        f'<span class="small" style="font-weight:400">{esc(NAMES.get(t, "").split()[-1] if NAMES.get(t) else "")}</span></span>'
        f'<span class="num{" fav" if fav(sd) else ""}">{num(sd)}</span></div>' for t, sd, col in ((a, "a", ca), (h, "h", ch)))
    bar = ""
    if p_home is not None:
        bar = (f'<div class="bar"><div style="width:{100 * (1 - p_home):.1f}%;background:{ca}"></div>'
               f'<div style="width:{100 * p_home:.1f}%;background:{ch}"></div></div>'
               f'<div class="meta">{p_lbl}: {a} {100 * (1 - p_home):.0f}% · {h} {100 * p_home:.0f}%</div>')
    meta = ""
    if sim and "p_home" in sim:
        meta = (f'<div class="meta">Projected {a} {sim["ga"]:.1f} – {sim["gh"]:.1f} {h} · most likely {esc(sim["likely"])} · '
                f'OT {100 * sim["ot"]:.0f}%</div><div class="meta">🥅 {esc(sim["ag"])} vs {esc(sim["hg"])}'
                + "</div>" + (f'<div class="meta">{esc(sim["fat"])}</div>' if sim.get("fat") else ""))
    elif sim and "error" in sim:
        meta = f'<div class="meta">Couldn\'t simulate: {esc(sim["error"])}</div>'
    with st.container(border=True):
        st.markdown(f'<div class="gc">{badge}{rows}{bar}{meta}</div>', unsafe_allow_html=True)
        if st.button("Open in Game Center ›", key=f"open_{g['id']}", width="stretch"):
            open_game(h, a, dict(id=g["id"], date=day_str, start=g.get("startTimeUTC", ""), playoff=g.get("gameType") == 3))
            st.switch_page(PAGE["game"])


def page_today():
    top = st.columns([3, 2, 2], vertical_alignment="bottom")
    top[0].markdown("## 📅 Games")
    day = top[1].date_input("Date", value=ss["day"], key="_w_day", on_change=_keep, args=("day",),
                            label_visibility="collapsed")
    with top[2]:
        settings_popover()
    day_str = day.isoformat()
    games = day_games_safe(day_str)
    if not games:
        st.info(f"No WHL games on {day:%A, %B %d}. Pick another date.")
        return
    sims = ss.get("slate", {}).get(day_str, {})
    c1, c2 = st.columns([2, 1], vertical_alignment="center")
    with c1:
        go_ = st.button(f"🎲 Simulate all {len(games)} games", type="primary", width="stretch",
                        help="Uses the best-guess starting goalies, each team's usual lineup, and travel & fatigue.")
    save = c2.toggle("Save to Tracker", value=True, key="slate_save",
                     help="Saves game AND player predictions so they're graded after the games. "
                          "Games that have started aren't changed.")
    if go_:
        n = min(ss["n_sims"], 3000)
        bar = st.progress(0.0, text="Simulating…")
        try:
            saved, saved_p = simulate_slate(day_str, games, n, save, bar)
            bar.empty()
            if save:
                st.toast(f"Saved {saved} game and {saved_p} player prediction set(s) to the Tracker." if saved or saved_p
                         else "Nothing new saved (those games have already started).")
        except Exception as e:
            st.error(f"Couldn't finish the slate ({e}).")
        sims = ss.get("slate", {}).get(day_str, {})

    any_live = False

    @st.fragment(run_every=60 if day_str == local_today().isoformat() else None)
    def cards():
        nonlocal any_live
        board = {g["id"]: g for g in scoreboard(day_str)}
        any_live = any(b.get("gameState") in ("LIVE", "CRIT") for b in board.values())
        for i in range(0, len(games), 3):
            cols = st.columns(3)
            for col, g in zip(cols, games[i:i + 3]):
                with col:
                    game_card(g, board.get(g["id"]), sims.get(g["id"]), day_str)
        if any_live:
            st.caption("🔴 Live scores and live win chances refresh every minute.")

    cards()
    picks_panel(day_str)

    if sims:
        st.markdown("#### 🔥 Top player projections")
        rows = [r for gid in sims for r in ss.get("slate_players", {}).get(day_str, {}).get(gid, []) if r["pos"] != "G"]
        if rows:
            df = pd.DataFrame(rows)
            pick = st.segmented_control("Sort by", ["Anytime goal", "1+ point", "3+ shots", "Points"], default="Anytime goal",
                                        key="today_sort")
            col = {"Anytime goal": "p_goal", "1+ point": "p_point", "3+ shots": "p_sog3", "Points": "proj_pts"}[pick or "Anytime goal"]
            df = df.sort_values(col, ascending=False).head(25)
            view = pd.DataFrame({"Player": df["player"], "Team": df["team"], "Opp": df["opp"],
                                 "Goal %": df["p_goal"] * 100, "Point %": df["p_point"] * 100,
                                 "3+ SOG %": df["p_sog3"] * 100, "Proj SOG": df["proj_sog"], "Proj PTS": df["proj_pts"],
                                 "Fair odds (goal)": [american(p) for p in df["p_goal"]]})
            st.dataframe(view, hide_index=True, width="stretch", height=420, column_config={
                "Goal %": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=60),
                "Point %": st.column_config.NumberColumn(format="%.0f%%"),
                "3+ SOG %": st.column_config.NumberColumn(format="%.0f%%"),
                "Proj SOG": st.column_config.NumberColumn(format="%.1f"),
                "Proj PTS": st.column_config.NumberColumn(format="%.2f")})
            st.caption("Full prop board with every stat and line is on the ⛸️ Players page.")


# ==========================================================================
# PAGE: Game Center
# ==========================================================================
def lineup_controls(prof, side, gs, cur_goalie, cur_out):
    g_def, g_status, g_note = gs
    st.markdown(f"**{_label(prof['team'])}**")
    gk = prof["goalies"]
    goalie = None
    if gk:
        ids = [g["id"] for g in gk]
        goalie = st.selectbox(
            "Starting goalie", ids, index=ids.index(cur_goalie) if cur_goalie in ids else 0,
            key=f"gk_{side}_{prof['team']}_{int(bool(ss['use_lines']))}",
            format_func=lambda i: next(f"{g['name']}  ({g['svp']:.3f} SV%, {g['gp']} GP)" for g in gk if g["id"] == i))
        st.caption(("✏️ Your pick" if goalie != g_def else GOALIE_ICON.get(g_status, g_status))
                   + (f" · {g_note}" if g_note else ""))
    else:
        st.caption("No goalie stats yet – using league average.")
    names = {s["id"]: f"{s['name']} ({s['pos']}, {s['g']}G {s['a']}A)" for s in prof["skaters"]}
    out = st.multiselect("Out tonight (injured / scratched)", list(names), default=[i for i in cur_out if i in names],
                         key=f"out_{side}_{prof['team']}_{int(bool(ss['use_lines']))}", format_func=lambda i: names[i],
                         help="Pre-filled with players who weren't in the last two lineups. Remove a name to bring "
                              "a player back; add a name to sit him.")
    return goalie, tuple(sorted(out))


def lines_html(prof):
    return ""


def headshots(*profs):
    return {p["id"]: p.get("headshot", "") for prof in profs for p in prof["skaters"] + prof["goalies"]}


def player_table(res, k, shots_map):
    b = res["props"][k]
    pts = b["G"].astype(np.int16) + b["A"]
    df = pd.DataFrame({
        "": [shots_map.get(i, "") for i in b["ids"]], "Player": b["names"], "Pos": b["pos"],
        "G": b["G"].mean(0), "A": b["A"].mean(0), "PTS": pts.mean(0), "SOG": b["SOG"].mean(0),
        "Goal %": (b["G"] > 0).mean(0) * 100, "Point %": (pts > 0).mean(0) * 100,
        "3+ SOG %": (b["SOG"] >= 3).mean(0) * 100, "PPP %": (b["PPP"] > 0).mean(0) * 100})
    return df.sort_values("PTS", ascending=False).reset_index(drop=True)


PLAYER_CFG = {"": st.column_config.ImageColumn(" ", width="small"),
              "G": st.column_config.NumberColumn(format="%.2f"), "A": st.column_config.NumberColumn(format="%.2f"),
              "PTS": st.column_config.NumberColumn(format="%.2f"), "SOG": st.column_config.NumberColumn(format="%.1f"),
              "Goal %": st.column_config.NumberColumn(format="%.0f%%"),
              "Point %": st.column_config.NumberColumn(format="%.0f%%"),
              "3+ SOG %": st.column_config.NumberColumn(format="%.0f%%"),
              "PPP %": st.column_config.NumberColumn(format="%.0f%%")}


def save_prediction(res, mi):
    ng = mi["ng"]
    evt = None
    if odds_api_key():
        try:
            evt = find_event(odds_feed(odds_api_key())[0], mi["home"], mi["away"])
        except Exception:
            evt = None
    hs = "Manual" if mi["hg"] != mi["gs"]["home"][0] else mi["gs"]["home"][1]
    as_ = "Manual" if mi["ag"] != mi["gs"]["away"][0] else mi["gs"]["away"][1]
    rec = make_record(res, ng, {"home": (goalie_name(mi["hp"], mi["hg"]), hs),
                                "away": (goalie_name(mi["ap"], mi["ag"]), as_)}, ss["n_recent"], "matchup", evt)
    n1 = upsert_predictions([rec], f"Save prediction {mi['away']} @ {mi['home']} {ng['date']}")
    n2 = upsert_players(player_records(res, dict(id=ng["id"], date=ng["date"], start=ng.get("start", ""))),
                        f"Save player predictions {mi['away']} @ {mi['home']} {ng['date']}")
    return n1 or n2


def page_game():
    c = st.columns([3, 3, 2], vertical_alignment="bottom")
    away = c[0].selectbox("Away", TEAMS, index=TEAMS.index(ss["away"]), key="_w_away", format_func=_label,
                          on_change=_keep, args=("away",))
    home = c[1].selectbox("Home", TEAMS, index=TEAMS.index(ss["home"]), key="_w_home", format_func=_label,
                          on_change=_keep, args=("home",))
    with c[2]:
        settings_popover()
    if home == away:
        st.warning("Pick two different teams.")
        return
    with st.spinner(f"Loading real play-by-play for {away} and {home}…"):
        try:
            mi = matchup_inputs(home, away)
        except Exception as e:
            st.error(f"Couldn't load team data ({e}).")
            return

    with st.expander("🔧 Lineups: starting goalies and injuries"):
        l1, l2 = st.columns(2)
        st.caption("No public source posts WHL line combinations, so the lineup is who dressed in the last two games.")
        with l1:
            ag, ao = lineup_controls(mi["ap"], "away", mi["gs"]["away"], mi["ag"], mi["ao"])
            st.markdown(lines_html(mi["ap"]), unsafe_allow_html=True)
        with l2:
            hg, ho = lineup_controls(mi["hp"], "home", mi["gs"]["home"], mi["hg"], mi["ho"])
            st.markdown(lines_html(mi["hp"]), unsafe_allow_html=True)
        ss[mi["key"]] = dict(hg=hg, ag=ag, ho=ho, ao=ao)
        mi.update(hg=hg, ag=ag, ho=ho, ao=ao)
        if st.button("↺ Back to the best-guess goalies & usual lineups", key="lu_reset"):
            ss.pop(mi["key"], None)
            for k in list(ss.keys()):
                if str(k).startswith(("gk_", "out_")):
                    del ss[k]
            st.rerun()

    with st.spinner(f"Simulating {ss['n_sims']:,} games…"):
        res = sim_for(mi)
    ng = mi["ng"]
    ch, ca = team_colors(home, away)
    a_st = "Manual" if mi["ag"] != mi["gs"]["away"][0] else mi["gs"]["away"][1]
    h_st = "Manual" if mi["hg"] != mi["gs"]["home"][0] else mi["gs"]["home"][1]
    when = (pd.Timestamp(ng["date"]).strftime("%a %b %d") + (f" · {local_time(ng['start'])}" if ng.get("start") else "")
            if ng else "No upcoming game between these teams")
    tired = fatigue_text(res.get("fatigue") or {}, home, away)
    team_hero(away, home, f"{when}<br>🥅 {esc(goalie_name(mi['ap'], mi['ag']))} {SHORT_ICON.get(a_st, '')} vs "
                          f"{esc(goalie_name(mi['hp'], mi['hg']))} {SHORT_ICON.get(h_st, '')}"
                          + (f"<br>{esc(tired)}" if tired else "")
                          + ("<br>🏆 Playoff rules" if res.get("playoff") else ""))
    s0 = res["top_scores"][0]
    tot = res["total"]
    kpis([(f"{away} win", pct1(res["away_win"]), f"fair {american(res['away_win'])}"),
          (f"{home} win", pct1(res["home_win"]), f"fair {american(res['home_win'])}"),
          ("Projected score", f"{res['goals_mean'][1]:.1f} – {res['goals_mean'][0]:.1f}", f"{away} – {home}"),
          ("Most likely final", f"{s0[0][1]} – {s0[0][0]}", f"{100 * s0[1]:.1f}% of sims"),
          ("Goes to OT/SO", pct1(res["ot"] + res["so"]), ""),
          ("Total goals", f"{tot.mean():.2f}", f"over 5.5: {100 * (tot > 5.5).mean():.0f}% · 6.5: {100 * (tot > 6.5).mean():.0f}%")])
    if ng:
        if st.button("📌 Save this game's prediction to the Tracker", key="save_match",
                     help="Saves the game AND player projections so they're graded after the game."):
            try:
                if save_prediction(res, mi):
                    st.success(f"Saved {away} @ {home} ({ng['date']}). The 📈 Tracker grades it after the game.")
                else:
                    st.info("That game has already started, so its saved prediction is locked.")
            except Exception as e:
                st.error(f"Couldn't save the prediction ({e}).")

    t_sum, t_watch, t_wi, t_play, t_odds = st.tabs(["📊 Summary", "🎬 Watch", "🧪 What-if", "⛸️ Players",
                                                     "💰 Odds & bets"])
    with t_wi:
        whatif_lab(mi, res)
    hp, ap = mi["hp"], mi["ap"]

    with t_sum:
        left, right = st.columns(2)
        with left:
            st.markdown("**Most common final scores**")
            top = res["top_scores"][:8]
            labels = [f"{away} {s[1]}–{s[0]} {home}" for s, _ in top]
            fig = go.Figure(go.Bar(x=[100 * p for _, p in top], y=labels, orientation="h",
                                   marker_color=[ch if s[0] > s[1] else ca for s, _ in top],
                                   text=[f"{100 * p:.1f}%" for _, p in top], textposition="outside", cliponaxis=False))
            fig.update_layout(yaxis=dict(autorange="reversed"), xaxis=dict(visible=False), margin=dict(t=4))
            chart(fig, "top_scores", 300)
        with right:
            st.markdown("**Goals per team**")
            fig = go.Figure()
            for i, (ab, col) in enumerate(((away, ca), (home, ch))):
                fig.add_trace(go.Bar(x=[str(k) if k < 9 else "9+" for k in range(10)], y=100 * res["goal_dist"][1 - i],
                                     name=ab, marker_color=col))
            fig.update_layout(barmode="group", yaxis_title="% of sims", margin=dict(t=4),
                              legend=dict(orientation="h", y=1.12, x=0))
            chart(fig, "goal_dist", 300)
        st.markdown("**Team comparison**")
        r_a, r_h = STN.loc[away], STN.loc[home]
        rows = [
            ("Win % (sim)", pct1(res["away_win"]), pct1(res["home_win"])),
            ("Regulation win %", pct1(res["away_reg_win"]), pct1(res["home_reg_win"])),
            ("Win by 2+ (puck line)", pct1(res["away_cover"]), pct1(res["home_cover"])),
            ("Goals (sim)", f"{res['goals_mean'][1]:.2f}", f"{res['goals_mean'][0]:.2f}"),
            ("Shots on goal (sim)", f"{res['sog_mean'][1]:.1f}", f"{res['sog_mean'][0]:.1f}"),
            ("PP goals / chances (sim)", f"{res['ppg_mean'][1]:.2f} / {res['ppo_mean'][1]:.2f}",
             f"{res['ppg_mean'][0]:.2f} / {res['ppo_mean'][0]:.2f}"),
            ("Goalie · sim SV%", f"{res['goalies'][1]['Goalie']} · {res['goalies'][1]['SV%']:.3f}",
             f"{res['goalies'][0]['Goalie']} · {res['goalies'][0]['SV%']:.3f}"),
            ("Record (W-L-OTL)", f"{r_a.w}-{r_a.l}-{r_a.otl}", f"{r_h.w}-{r_h.l}-{r_h.otl}"),
            ("GF / GA per game", f"{r_a.gf / max(r_a.gp, 1):.2f} / {r_a.ga / max(r_a.gp, 1):.2f}",
             f"{r_h.gf / max(r_h.gp, 1):.2f} / {r_h.ga / max(r_h.gp, 1):.2f}"),
            ("Last 10 · streak", f"{r_a.l10} · {r_a.streak}", f"{r_h.l10} · {r_h.streak}"),
            ("Recent SOG for / against", f"{ap['sog_pg']:.1f} / {ap['soga_pg']:.1f}", f"{hp['sog_pg']:.1f} / {hp['soga_pg']:.1f}"),
            ("Recent PP% / PK%", f"{pct(ap['pp_pct'])} / {pct(ap['pk_pct'])}", f"{pct(hp['pp_pct'])} / {pct(hp['pk_pct'])}"),
            ("Faceoff %", pct(ap["fo_pct"]), pct(hp["fo_pct"])),
            ("Lineup strength vs usual", f"{100 * (res['sides']['away']['off'] - 1):+.0f}%",
             f"{100 * (res['sides']['home']['off'] - 1):+.0f}%"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["", away, home]), hide_index=True, width="stretch")
        if st.button(f"⚔️ See the {away}–{home} rivalry", key="gc_riv"):
            ss["riv_a"], ss["riv_b"] = away, home
            st.switch_page(PAGE["rivalry"])
        st.markdown("**When goals happen: simulation vs. real recent games**")
        fig = go.Figure()
        for i, (ab, prof, col) in enumerate(((home, hp, ch), (away, ap, ca))):
            fig.add_trace(go.Bar(x=["1st", "2nd", "3rd", "OT"], y=res["goals_by_per"][i], name=f"{ab} sim", marker_color=col))
            fig.add_trace(go.Bar(x=["1st", "2nd", "3rd", "OT"], y=prof["gf_per"], name=f"{ab} real",
                                 marker=dict(color=col, pattern_shape="/", opacity=0.5)))
        fig.update_layout(barmode="group", yaxis_title="Goals / game", margin=dict(t=4),
                          legend=dict(orientation="h", y=1.15, x=0))
        chart(fig, "per_goals", 290)

    with t_watch:
        f = res["featured"]
        fs = f["final"]
        st.markdown(f"**A full simulated game with the most common result: {away} {fs[1]} – {fs[0]} {home}"
                    f"{' (OT)' if f['ot'] else ' (SO)' if f['so'] else ''}**")
        st.caption("Tap ▶ to play, with play-by-play calls and jersey numbers. Drag along the win-probability strip "
                   "to jump anywhere; Goal ⏩ skips to the next goal. "
                   "🎥 switches between a camera that follows the puck and the full rink.")
        if hasattr(st, "iframe"):
            st.iframe(animation_html(res), height="content")
        else:
            components.html(animation_html(res), height=900, scrolling=False)
        with st.expander("📈 Game flow (shots, goals and power plays)"):
            chart(flow_figure(res), "flow")
        with st.expander("🧾 Scoring summary & penalties"):
            log_rows = [dict(Time=clock(e[1]) if e[3] != "so" else "Shootout", Play=event_text(e, f["names"], home, away),
                             Score=f"{away} {e[10]} – {e[9]} {home}") for e in f["events"] if e[3] in ("goal", "pen", "so")]
            st.dataframe(pd.DataFrame(log_rows), hide_index=True, width="stretch")

    with t_play:
        shots_map = headshots(hp, ap)
        st.caption("Average per simulated game, plus the chance of each milestone. Tap a column to sort.")
        for k, ab in ((1, away), (0, home)):
            st.markdown(f"**{_label(ab)}**")
            st.dataframe(player_table(res, k, shots_map), hide_index=True, width="stretch", height=360,
                         column_config=PLAYER_CFG)
            gk = res["goalies"][k]
            st.caption(f"🥅 {gk['Goalie']}: {gk['Saves']:.1f} saves · {gk['GA']:.2f} GA · {gk['SV%']:.3f} SV% · "
                       f"shutout {gk['Shutout %']:.1f}%")
        st.divider()
        player_card_ui(res, hp, ap, "gc")

    with t_odds:
        game_odds_ui(res, ng, mi)


# ==========================================================================
# Players: prop probabilities, player cards with real game logs
# ==========================================================================
SK_STATS = ["SOG", "Points", "Goals", "Assists", "PP points"]
LINES = {"SOG": [1.5, 2.5, 3.5, 4.5], "Points": [0.5, 1.5, 2.5], "Goals": [0.5, 1.5], "Assists": [0.5, 1.5],
         "PP points": [0.5, 1.5]}


def saves_lines(arr):
    m = float(np.median(arr))
    base = math.floor(m) + 0.5
    return [base - 4, base - 2, base, base + 2, base + 4]


def player_options(res):
    opts = []
    for k, b in enumerate(res["props"]):
        pts = (b["G"].astype(np.int16) + b["A"]).mean(0)
        for i in np.argsort(-pts):
            opts.append(("S", k, int(i)))
        if b.get("goalie_id"):
            opts.append(("G", k, -1))
    return opts


def player_card_ui(res, hp, ap, where, preselect=None):
    st.markdown("#### 🔎 Player card")
    opts = player_options(res)
    if not opts:
        return

    def fmt(o):
        kind, k, i = o
        b = res["props"][k]
        return f"{b['goalie']} ({b['team']}, G)" if kind == "G" else f"{b['names'][i]} ({b['team']}, {b['pos'][i]})"

    idx = opts.index(preselect) if preselect in opts else 0
    c1, c2 = st.columns([3, 2])
    o = c1.selectbox("Player", opts, index=idx, format_func=fmt, key=f"pc_player_{where}")
    kind, k, i = o
    b = res["props"][k]
    if kind == "G":
        stat = "Saves"
        c2.markdown("Stat: **Saves**")
        arr, pid, name, pos = b["saves"], b.get("goalie_id"), b["goalie"], "G"
    else:
        stat = c2.segmented_control("Stat", SK_STATS, default="SOG", key=f"pc_stat_{where}") or "SOG"
        arr, pid, name, pos = stat_array(b, i, stat), b["ids"][i], b["names"][i], b["pos"][i]
    arr = np.asarray(arr)
    lines = saves_lines(arr) if stat == "Saves" else LINES[stat]
    default_line = min(lines, key=lambda L: abs((arr > L).mean() - 0.5))
    line = st.select_slider("Line", lines, value=default_line, key=f"pc_line_{where}_{stat}",
                            format_func=lambda v: f"{v:g}")
    prof = hp if hp["team"] == b["team"] else ap
    info = next((p for p in prof["skaters"] + prof["goalies"] if p["id"] == pid), {})
    if pos == "G":
        season = (f"{info.get('gp', 0)} GP · {info.get('w', 0)} W · {info.get('svp', 0):.3f} SV% · "
                  f"{info.get('gaa', 0):.2f} GAA") if info else ""
    else:
        gp = max(info.get("gp", 1), 1)
        season = (f"{info.get('gp', 0)} GP · {info.get('g', 0)} G · {info.get('a', 0)} A · {info.get('pts', 0)} PTS · "
                  f"{info.get('shots', 0) / gp:.1f} SOG/GP") if info else ""
    opp = res["props"][1 - k]["team"]
    st.markdown(f'<div class="pc"><img src="{esc(info.get("headshot", ""))}" alt=""><div><div class="nm">{esc(name)}</div>'
                f'<div class="sub">{b["team"]} · {pos} · vs {opp}</div><div class="sub">This season: {esc(season)}</div>'
                "</div></div>", unsafe_allow_html=True)
    p_over, _ = over_under(arr, line, "Over")
    p_under, _ = over_under(arr, line, "Under")
    recent = recent_values(pid, stat, team=b["team"]) if pid else []
    hits = sum(v > line for _, _, v in recent)
    kpis([("Projection", f"{arr.mean():.2f}", f"median {np.median(arr):g}"),
          (f"Over {line:g}", pct1(p_over), f"fair {american(p_over)}"),
          (f"Under {line:g}", pct1(p_under), f"fair {american(p_under)}"),
          ("Last 10 games", f"{hits}/{len(recent)} over" if recent else "–",
           f"avg {np.mean([v for *_, v in recent]):.1f}" if recent else "no game log")])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Simulated outcomes**")
        hi = int(np.percentile(arr, 99.5)) if len(arr) else 0
        lo = int(arr.min()) if stat == "Saves" else 0
        xs = list(range(lo, max(hi, int(line) + 2) + 1))
        ps = [100 * float((arr == x).mean()) for x in xs]
        fig = go.Figure(go.Bar(x=xs, y=ps, marker_color=["#27ae60" if x > line else "#b0b8c8" for x in xs],
                               hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
        fig.add_vline(x=line, line=dict(color="#c0392b", dash="dot"))
        fig.update_layout(xaxis_title=stat, yaxis_title="% of sims", margin=dict(t=6), bargap=0.15)
        chart(fig, f"pc_dist_{where}", 260)
    with c2:
        st.markdown("**Last 10 real games**")
        if recent:
            rv = recent[::-1]
            fig = go.Figure(go.Bar(x=[f"{d[5:]} {o_}" for d, o_, _ in rv], y=[v for *_, v in rv],
                                   marker_color=["#27ae60" if v > line else "#c0392b" for *_, v in rv],
                                   text=[v for *_, v in rv], textposition="outside", cliponaxis=False))
            fig.add_hline(y=line, line=dict(color="#8ea0c4", dash="dot"))
            fig.update_layout(yaxis_title=stat, margin=dict(t=6), xaxis=dict(tickangle=-45, tickfont=dict(size=10)))
            chart(fig, f"pc_log_{where}", 260)
        else:
            st.caption("The WHL doesn't publish this stat game by game." if stat == "PP points"
                       else "No game log available for this player yet.")


def prop_board(results, stat, min_prob=0):
    """One row per player across the given simulated games, with chances for each line."""
    rows = []
    for res in results:
        for k, b in enumerate(res["props"]):
            opp = res["props"][1 - k]["team"]
            if stat == "Saves":
                if not b.get("goalie_id"):
                    continue
                arr = np.asarray(b["saves"])
                ls = saves_lines(arr)
                r = {"Player": b["goalie"], "Team": b["team"], "Opp": opp, "Proj": arr.mean()}
                for j, L in enumerate(ls):
                    r[f"Line {j + 1}"] = f"{L:g}: {100 * (arr > L).mean():.0f}%"
                rows.append(r)
                continue
            for i in range(len(b["ids"])):
                arr = np.asarray(stat_array(b, i, stat))
                r = {"Player": b["names"][i], "Team": b["team"], "Opp": opp, "Pos": b["pos"][i], "Proj": arr.mean()}
                for L in LINES[stat]:
                    r[f"Over {L:g}"] = 100 * (arr > L).mean()
                main = LINES[stat][0] if stat in ("Goals", "Points", "Assists", "PP points") else 2.5
                r[f"Fair odds o{main:g}"] = american(float((arr > main).mean()))
                rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    first = [c for c in df.columns if c.startswith("Over ")]
    if first and min_prob:
        df = df[df[first[0]] >= min_prob]
    return df.sort_values("Proj", ascending=False).reset_index(drop=True)


def page_players():
    top = st.columns([3, 2], vertical_alignment="bottom")
    top[0].markdown("## ⛸️ Players & props")
    with top[1]:
        settings_popover()
    day_str = ss["day"].isoformat()
    slate = ss.get("slate", {}).get(day_str, {})
    srcs = [f"{ss['away']} @ {ss['home']} (Game Center)"]
    if slate:
        srcs.append(f"Every simulated game on {ss['day']:%b %d}")
    src = st.segmented_control("Games", srcs, default=srcs[-1], key="pl_src") or srcs[0]
    results, res_by_label = [], {}
    if src.startswith("Every") and slate:
        n = min(ss["n_sims"], 3000)
        for g in day_games_safe(day_str):
            if g["id"] in slate and "error" not in slate[g["id"]]:
                try:
                    r = sim_game(g, day_str, n)
                    results.append(r[0])
                    res_by_label[f"{r[0]['away']} @ {r[0]['home']}"] = r
                except Exception:
                    pass
    else:
        if ss["home"] == ss["away"]:
            st.warning("Pick two different teams in the Game Center.")
            return
        mi = matchup_inputs(ss["home"], ss["away"])
        with st.spinner("Simulating…"):
            r = sim_for(mi)
        results = [r]
        res_by_label[f"{r['away']} @ {r['home']}"] = (r, None, mi["hp"], mi["ap"])
        if not slate:
            st.caption("Tip: simulate the whole day on the 📅 Games page to see every player playing that day.")
    if not results:
        st.info("No simulated games yet.")
        return
    c1, c2 = st.columns([3, 2])
    stat = c1.segmented_control("Stat", SK_STATS + ["Saves"], default="SOG", key="pl_stat") or "SOG"
    min_prob = c2.slider("Hide players under this chance (first line)", 0, 60, 0, 5, key="pl_min",
                         format="%d%%") if stat != "Saves" else 0
    board = prop_board(results, stat, min_prob)
    if board.empty:
        st.info("No players to show.")
    else:
        cfg = {c: st.column_config.NumberColumn(format="%.0f%%") for c in board.columns if c.startswith("Over ")}
        cfg["Proj"] = st.column_config.NumberColumn(format="%.2f")
        st.dataframe(board, hide_index=True, width="stretch", height=480, column_config=cfg)
        st.caption("Chance the player goes OVER each line in the simulations. Fair odds have no sportsbook margin; "
                   "a sportsbook price better than fair odds is a value bet if the model is right.")
    st.divider()
    lab = st.selectbox("Player card: game", list(res_by_label), key="pl_card_game")
    r, _, hp_, ap_ = res_by_label[lab]
    if hp_ is None:
        hp_, ap_ = build_profile(r["home"], ss["n_recent"]), build_profile(r["away"], ss["n_recent"])
    player_card_ui(r, hp_, ap_, "pl")


# ==========================================================================
# Odds and logging bets
# ==========================================================================
def bet_row(res, game, market, team=None, side=None, line=None, player_id=None, player=None, stat=None,
            odds=-110, stake=10.0, book="", source="manual"):
    p, push = model_prob(res, market, team=team, side=side, line=line, player_id=player_id, stat=stat, player=player)
    r = dict(date=game["date"], game_id=int(game["id"]), start_utc=game.get("start", ""),
             game=f"{res['away']} @ {res['home']}", market=market, team=team, player_id=player_id, player=player,
             stat=stat, side=side, line=line, odds=int(odds), stake=float(stake),
             model_p=None if p is None else round(p, 4), book=book, source=source)
    r["selection"] = bet_label(r)
    return r, p, push


def ev_per100(p, push, price):
    if p is None:
        return None
    return p * payout(price) - (1 - p - (push or 0)) * 100


def leg_form(res, where):
    """Pick one bet on this game (market, team/side, line, player). Returns a leg dict."""
    home, away = res["home"], res["away"]
    mk = st.segmented_control("Market", ["Moneyline", "Puck line", "Total", "Player prop"], default="Moneyline",
                              key=f"bl_mk_{where}") or "Moneyline"
    c1, c2, c3 = st.columns(3)
    kw = {}
    if mk == "Moneyline":
        kw = dict(market="Moneyline", team=c1.radio("Team", [away, home], horizontal=True, key=f"bl_t_{where}"))
    elif mk == "Puck line":
        t = c1.radio("Team", [away, home], horizontal=True, key=f"bl_t_{where}")
        ln = c2.select_slider("Line", [-2.5, -1.5, 1.5, 2.5], value=1.5 if t == away else -1.5, key=f"bl_pl_{where}",
                              format_func=lambda v: f"{v:+g}")
        kw = dict(market="Puck line", team=t, line=ln)
    elif mk == "Total":
        kw = dict(market="Total", side=c1.radio("Side", ["Over", "Under"], horizontal=True, key=f"bl_s_{where}"),
                  line=c2.number_input("Goals line", 3.5, 9.5, 6.5, 0.5, key=f"bl_tl_{where}"))
    else:
        opts = player_options(res)

        def fmt(o):
            kind, k, i = o
            b = res["props"][k]
            return f"{b['goalie']} ({b['team']}, G)" if kind == "G" else f"{b['names'][i]} ({b['team']}, {b['pos'][i]})"

        o = c1.selectbox("Player", opts, format_func=fmt, key=f"bl_p_{where}")
        kind, k, i = o
        b = res["props"][k]
        stat = "Saves" if kind == "G" else c2.selectbox("Stat", SK_STATS, key=f"bl_st_{where}")
        side = c3.radio("Side", ["Over", "Under"], horizontal=True, key=f"bl_s2_{where}")
        dflt = 24.5 if stat == "Saves" else (2.5 if stat == "SOG" else 0.5)
        ln = c3.number_input("Line", 0.5, 60.5, dflt, 1.0, key=f"bl_l_{where}_{stat}")
        pid = b.get("goalie_id") if kind == "G" else b["ids"][i]
        kw = dict(market="Player", side=side, line=float(ln), stat=stat,
                  player_id=None if pid is None else int(pid),
                  player=b["goalie"] if kind == "G" else b["names"][i], team=b["team"])
    return kw


def bet_logger(res, game, where):
    """Log a bet you placed (or are thinking about) and see what the model thinks of it."""
    if not game:
        st.info("This matchup isn't on the schedule, so bets on it can't be graded.")
        return
    kw = leg_form(res, where)
    d1, d2, d3 = st.columns(3)
    odds = d1.number_input("Odds (American)", -10000, 10000, -110, 5, key=f"bl_o_{where}")
    stake = d2.number_input("Stake ($)", 0.0, 100000.0, 10.0, 5.0, key=f"bl_k_{where}")
    book = d3.text_input("Sportsbook (optional)", key=f"bl_b_{where}")
    if -100 < odds < 100:
        st.warning("American odds are +100 or higher, or -100 or lower.")
        return
    row, p, push = bet_row(res, game, odds=odds, stake=stake, book=book, **kw)
    if p is None:
        st.caption("The model can't price this one (player isn't in the simulated lineup).")
    else:
        ev = ev_per100(p, push, odds)
        kel = kelly(p, odds) * 0.25 * ss["bankroll"]
        kpis([("Model chance", pct1(p), f"fair odds {american(p)}"), ("Book's implied", pct1(implied(odds)), "with margin"),
              ("Expected profit", money(ev), "per $100 bet", "up" if ev > 0 else "down"),
              ("¼-Kelly stake", f"${kel:,.0f}", f"bankroll ${ss['bankroll']:,.0f}")])
        cls, word = (("good", "🟢 Great value") if ev >= 8 else ("good", "🟢 Some value") if ev > 2 else
                     ("ok", "⚪ About a fair price") if ev > -4 else ("bad", "🔴 Poor value"))
        st.markdown(f'<div class="verdict {cls}">{word}: the model gives this {pct1(p)}; odds of {fmt_odds(odds)} '
                    f'need {pct1(implied(odds))} to break even.</div>', unsafe_allow_html=True)
    if st.button(f"➕ Add “{row['selection']}” to My Bets", type="primary", key=f"bl_add_{where}"):
        try:
            add_bets([row])
            st.success("Logged. It will be graded automatically after the game (📈 on the 💰 Bets page).")
        except Exception as e:
            st.error(f"Couldn't save the bet ({e}).")


def best_combos(res, top=8):
    """Two-leg combos where the legs help each other: a team winning + one of its players getting a point
    (or a goal), and the total going over + a player's point."""
    rows = []
    m = res["margin"]
    for k, b in enumerate(res["props"]):
        team = b["team"]
        win = (m > 0) if team == res["home"] else (m < 0)
        pts = b["G"].astype(np.int16) + b["A"]
        for i in np.argsort(-pts.mean(0))[:5]:
            for stat, arr in (("Points", pts[:, i] > 0), ("Goals", b["G"][:, i] > 0)):
                joint = float((win & arr).mean())
                indep = float(win.mean() * arr.mean())
                if joint > 0.05:
                    rows.append({"Combo": f"{team} win + {b['names'][i]} {'1+ point' if stat == 'Points' else 'to score'}",
                                 "Chance": 100 * joint, "Fair odds": american(joint), "If unrelated": american(indep),
                                 "Boost": joint / indep if indep else 1.0,
                                 "_legs": [dict(market="Moneyline", team=team),
                                           dict(market="Player", player_id=int(b["ids"][i]), player=b["names"][i],
                                                stat=stat, side="Over", line=0.5, team=team)]})
    rows.sort(key=lambda r: -r["Boost"] * r["Chance"])
    return rows[:top]


def parlay_builder(res, game, where):
    """Same-game parlay: legs are priced together from the same simulated games."""
    key = f"parlay_{where}_{res['away']}_{res['home']}"
    legs = ss.setdefault(key, [])
    st.caption("Adds up the chance that every leg hits in the same simulated game, so legs that go together "
               "(a team winning and its top scorer getting a point) are priced correctly instead of just multiplied.")
    combos = best_combos(res)
    if combos:
        with st.expander("💡 Combos where the legs help each other"):
            view = pd.DataFrame([{k: v for k, v in c.items() if k != "_legs"} for c in combos])
            ev_ = st.dataframe(view, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                               key=f"combo_{key}", column_config={
                                   "Chance": st.column_config.NumberColumn(format="%.1f%%"),
                                   "Boost": st.column_config.NumberColumn("Boost vs unrelated", format="%.2f×")})
            sel = ev_.selection.rows if ev_ and hasattr(ev_, "selection") else []
            if sel and st.button("Load this combo into the parlay", key=f"combo_load_{key}"):
                ss[key] = [dict(l) for l in combos[sel[0]]["_legs"]]
                st.rerun()
    with st.container(border=True):
        st.markdown("**Add a leg**")
        leg = leg_form(res, f"{where}_pl")
        c1, c2 = st.columns(2)
        if c1.button("➕ Add leg", key=f"pl_add_{key}", width="stretch"):
            if leg not in legs:
                legs.append(leg)
            st.rerun()
        if legs and c2.button("🗑️ Clear parlay", key=f"pl_clear_{key}", width="stretch"):
            ss[key] = []
            st.rerun()
    if not legs:
        st.caption("Add two or more legs to price a parlay.")
        return
    pr = parlay_prob(res, legs)
    if pr is None:
        st.warning("One of the legs can't be priced (player isn't in the simulated lineup). Clear it and try again.")
        return
    slip = "".join(f'<div class="leg"><span>{esc(bet_label(dict(l, game=f"{res["away"]} @ {res["home"]}")))}</span>'
                   f'<span>{pct1(p)}</span></div>' for l, p in zip(legs, pr["legs"]))
    st.markdown(f'<div class="slip"><div class="small" style="margin-bottom:4px">🧾 Parlay · {len(legs)} legs</div>'
                f'{slip}</div>', unsafe_allow_html=True)
    rm = st.pills("Remove a leg", [f"{i + 1}" for i in range(len(legs))], key=f"pl_rm_{key}_{len(legs)}",
                  label_visibility="collapsed") if len(legs) > 1 else None
    if rm:
        legs.pop(int(rm) - 1)
        st.rerun()
    boost = pr["joint"] / pr["indep"] if pr["indep"] else 1.0
    kpis([("Parlay hits", pct1(pr["joint"]), f"fair odds {american(pr['joint'])}", "up" if boost > 1.05 else ""),
          ("If legs were unrelated", pct1(pr["indep"]), f"fair odds {american(pr['indep'])}"),
          ("Legs work together", f"{boost:.2f}×", "above 1 = they help each other", "up" if boost > 1.05 else
           ("down" if boost < 0.95 else ""))])
    if not game:
        st.info("This matchup isn't on the schedule, so a parlay on it can't be graded.")
        return
    d1, d2, d3 = st.columns(3)
    odds = d1.number_input("Sportsbook's parlay odds", -10000, 100000, 300, 10, key=f"pl_o_{key}")
    stake = d2.number_input("Stake ($)", 0.0, 100000.0, 10.0, 5.0, key=f"pl_k_{key}")
    book = d3.text_input("Sportsbook (optional)", key=f"pl_b_{key}")
    if -100 < odds < 100:
        st.warning("American odds are +100 or higher, or -100 or lower.")
        return
    ev = ev_per100(pr["joint"], pr["push"], odds)
    kpis([("Book's implied", pct1(implied(odds)), "with margin"),
          ("Expected profit", money(ev), "per $100 bet", "up" if ev > 0 else "down"),
          ("¼-Kelly stake", f"${kelly(pr['joint'], odds) * 0.25 * ss['bankroll']:,.0f}", f"bankroll ${ss['bankroll']:,.0f}")])
    clean = [{k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
              for k, v in l.items()} for l in legs]
    row = dict(date=game["date"], game_id=int(game["id"]), start_utc=game.get("start", ""),
               game=f"{away_home(res)}", market="Parlay", legs=json.dumps(clean), odds=int(odds), stake=float(stake),
               model_p=round(pr["joint"], 4), book=book, source="parlay")
    row["selection"] = bet_label(row)
    if st.button(f"➕ Add this {len(legs)}-leg parlay to My Bets", type="primary", key=f"pl_log_{key}"):
        try:
            add_bets([row])
            ss[key] = []
            st.success("Parlay logged. It's graded leg by leg after the game.")
        except Exception as e:
            st.error(f"Couldn't save the parlay ({e}).")


def away_home(res):
    return f"{res['away']} @ {res['home']}"


def edge_table(rows, where, game):
    """Editable table: tick bets to log them."""
    if not rows:
        return
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_row"} for r in rows])
    df.insert(0, "Log", False)
    df["Stake"] = 10.0
    show = ["Log", "Bet", "Book", "Odds", "Book chance", "Model chance", "Edge", "EV/$100", "Stake"]
    cfg = {"Log": st.column_config.CheckboxColumn("Log", width="small"),
           "Book chance": st.column_config.NumberColumn("Book (no vig)", format="%.1f%%"),
           "Model chance": st.column_config.NumberColumn(format="%.1f%%"),
           "Edge": st.column_config.NumberColumn(format="%+.1f%%"),
           "EV/$100": st.column_config.NumberColumn("EV per $100", format="%+.1f"),
           "Stake": st.column_config.NumberColumn(format="$%.0f", min_value=0.0, step=5.0)}
    ed = st.data_editor(df, column_order=show, hide_index=True, width="stretch", column_config=cfg,
                        disabled=[c for c in show if c not in ("Log", "Stake")], key=f"edge_{where}", height=380)
    pick = ed[ed["Log"]]
    if len(pick) and st.button(f"➕ Log {len(pick)} ticked bet(s)", type="primary", key=f"edge_add_{where}"):
        out = []
        for ix, r in pick.iterrows():
            row = dict(rows[ix]["_row"])
            row["stake"] = float(r["Stake"])
            out.append(row)
        try:
            add_bets(out)
            st.success(f"Logged {len(out)} bet(s).")
        except Exception as e:
            st.error(f"Couldn't save ({e}).")


def game_line_rows(res, evt, game):
    rows = []
    home, away = res["home"], res["away"]
    for bk in evt.get("bookmakers", []):
        for mk in bk.get("markets", []):
            outs = mk.get("outcomes", [])
            for oc in outs:
                price, pt = oc.get("price"), oc.get("point")
                if price is None:
                    continue
                if mk["key"] == "h2h":
                    t = home if same_team(oc["name"], home) else away
                    kw = dict(market="Moneyline", team=t)
                    pair = outs
                elif mk["key"] == "spreads" and pt is not None:
                    t = home if same_team(oc["name"], home) else away
                    kw = dict(market="Puck line", team=t, line=float(pt))
                    pair = [x for x in outs if abs(x.get("point") or 0) == abs(pt)]
                elif mk["key"] == "totals" and pt is not None:
                    kw = dict(market="Total", side=oc["name"], line=float(pt))
                    pair = [x for x in outs if x.get("point") == pt]
                else:
                    continue
                row, p, push = bet_row(res, game, odds=price, book=bk.get("title", ""), source="edge", **kw)
                if p is None:
                    continue
                novig = implied(price) / sum(implied(x["price"]) for x in pair if x.get("price"))
                rows.append({"Bet": row["selection"], "Book": bk.get("title", ""), "Odds": f"{int(price):+d}",
                             "Book chance": novig * 100, "Model chance": p * 100, "Edge": (p - novig) * 100,
                             "EV/$100": ev_per100(p, push, price), "_row": row})
    best = {}
    for r in rows:
        if r["Bet"] not in best or r["EV/$100"] > best[r["Bet"]]["EV/$100"]:
            best[r["Bet"]] = r
    return sorted(best.values(), key=lambda r: -r["EV/$100"])


def prop_rows(res, props_js, game):
    lines = prop_lines(props_js)
    pairs = {}
    for r in lines:
        pairs.setdefault((r["book"], _norm(r["player"]), r["stat"], r["line"]), {})[r["side"]] = r["price"]
    out = {}
    for r in lines:
        b, i = find_player(res, None, r["player"]) if r["stat"] != "Saves" else (find_goalie(res, None, r["player"]), None)
        if b is None:
            continue
        pid = b.get("goalie_id") if r["stat"] == "Saves" else b["ids"][i]
        name = b["goalie"] if r["stat"] == "Saves" else b["names"][i]
        side = "Over" if r["side"] in ("Over", "Yes") else "Under"
        row, p, push = bet_row(res, game, market="Player", team=b["team"], side=side, line=r["line"], player_id=pid,
                               player=name, stat=r["stat"], odds=r["price"], book=r["book"], source="edge")
        if p is None:
            continue
        pr = pairs.get((r["book"], _norm(r["player"]), r["stat"], r["line"]), {})
        if len(pr) == 2:
            novig = implied(r["price"]) / sum(implied(v) for v in pr.values())
        else:
            novig = implied(r["price"])
        cand = {"Bet": row["selection"], "Book": r["book"], "Odds": f"{r['price']:+d}", "Book chance": novig * 100,
                "Model chance": p * 100, "Edge": (p - novig) * 100, "EV/$100": ev_per100(p, push, r["price"]), "_row": row}
        if row["selection"] not in out or cand["EV/$100"] > out[row["selection"]]["EV/$100"]:
            out[row["selection"]] = cand
    return sorted(out.values(), key=lambda r: -r["EV/$100"])


def game_odds_ui(res, ng, mi):
    home, away = res["home"], res["away"]
    m_, t_ = res["margin"], res["total"]
    fair = [(f"{away} moneyline", res["away_win"]), (f"{home} moneyline", res["home_win"]),
            (f"{away} +1.5", (m_ <= 1).mean()), (f"{home} -1.5", (m_ >= 2).mean()),
            (f"{away} -1.5", (m_ <= -2).mean()), (f"{home} +1.5", (m_ >= -1).mean())]
    for line in (5.5, 6.5):
        fair += [(f"Over {line}", (t_ > line).mean()), (f"Under {line}", (t_ < line).mean())]
    st.markdown("**Fair odds from the simulation** (no sportsbook margin)")
    st.dataframe(pd.DataFrame([{"Bet": b, "Model chance": 100 * p, "Fair odds": american(p)} for b, p in fair]),
                 hide_index=True, width="stretch",
                 column_config={"Model chance": st.column_config.NumberColumn(format="%.1f%%")})
    game = dict(id=ng["id"], date=ng["date"], start=ng.get("start", "")) if ng else None
    key = odds_key_input("gc")
    if key:
        b1, b2 = st.columns(2)
        if b1.button("Get game lines (3 credits)", key="gc_lines", width="stretch"):
            ss["_gc_lines_on"] = True
        if b2.button("Get player props (~6 credits)", key="gc_props", width="stretch"):
            ss["_gc_props_on"] = True
        evt = None
        if ss.get("_gc_lines_on") or ss.get("_gc_props_on"):
            try:
                feed, left = odds_feed(key)
                evt = find_event(feed, home, away)
                st.caption(f"API credits left this month: {left}")
            except Exception as e:
                st.error(f"Couldn't get odds ({e}). Check the API key.")
            if evt is None:
                st.info(f"No sportsbook lines for {away} @ {home} yet. Books usually post them a day or two ahead.")
        if evt and game:
            if ss.get("_gc_lines_on"):
                st.markdown("**Game lines: best price for each bet** (tick to log it)")
                edge_table(game_line_rows(res, evt, game), "gc_lines", game)
            if ss.get("_gc_props_on"):
                try:
                    pj, left = prop_feed(key, evt["id"])
                    rows = prop_rows(res, pj, game)
                    st.markdown(f"**Player props: best price for each bet** ({len(rows)} matched)")
                    if rows:
                        edge_table(rows, "gc_props", game)
                    else:
                        st.info("No player props posted for this game yet.")
                except Exception as e:
                    st.error(f"Couldn't get player props ({e}).")
    st.markdown("**✍️ Check a line you've seen and log your bet**")
    st.caption("Type in the odds from your sportsbook and the app tells you whether it's good value.")
    bet_logger(res, game, "gc")
    st.markdown("#### 🧩 Same-game parlay builder")
    parlay_builder(res, game, "gc")


# ==========================================================================
# PAGE: Bets (your own bets, graded automatically)
# ==========================================================================
def market_group(r):
    return f"Prop · {r['stat']}" if r["market"] == "Player" else r["market"]


def page_bets():
    top = st.columns([3, 2], vertical_alignment="bottom")
    top[0].markdown("## 💰 My bets")
    top[1].number_input("Bankroll ($)", 0.0, 1e7, float(ss["bankroll"]), 50.0, key="_w_bankroll", on_change=_keep,
                        args=("bankroll",), help="Used for the ¼-Kelly stake suggestions.")
    try:
        bets, newly = grade_bets()
        if newly:
            st.toast(f"Graded {newly} bet(s).")
    except Exception as e:
        bets = load_store("bets")
        st.error(f"Couldn't grade bets ({e}). If you use GitHub storage, check the token and repo name.")
    bets = bets.copy()
    for c in ("stake", "profit", "odds", "model_p", "line"):
        bets[c] = pd.to_numeric(bets[c], errors="coerce")
    openb = bets[bets["status"] == "Open"]
    done = bets[bets["status"].isin(["Won", "Lost", "Push"])]
    w, l, p_ = (done["status"] == "Won").sum(), (done["status"] == "Lost").sum(), (done["status"] == "Push").sum()
    staked = float(done["stake"].sum())
    profit = float(done["profit"].sum())
    kpis([("Open bets", f"{len(openb)}", f"${openb['stake'].sum():,.0f} at risk"),
          ("Record", f"{w}-{l}" + (f"-{p_}" if p_ else ""), f"{100 * w / max(w + l, 1):.0f}% won"),
          ("Profit", money(profit), f"on ${staked:,.0f} staked"),
          ("ROI", f"{100 * profit / staked:+.1f}%" if staked else "–", "profit ÷ staked"),
          ("Model said", pct1(done["model_p"].mean()) if done["model_p"].notna().any() else "–", "avg chance on settled bets")])
    t_log, t_list, t_res = st.tabs(["➕ Log a bet", "📋 My bets", "📈 Results"])
    t_edge = st.container()

    with t_log:
        day_str = ss["day"].isoformat()
        choices = todays_matchups(day_str)
        labels = [lab for lab, _ in choices]
        gc_label = f"{ss['away']} @ {ss['home']} (Game Center matchup)"
        pick = st.selectbox(f"Game ({ss['day']:%b %d} — change the date on 📅 Games)", labels + [gc_label], key="bp_game")
        try:
            if pick == gc_label:
                mi = matchup_inputs(ss["home"], ss["away"])
                res = sim_for(mi)
                ng = mi["ng"]
                game = dict(id=ng["id"], date=ng["date"], start=ng.get("start", "")) if ng else None
            else:
                g = dict(choices)[pick]
                with st.spinner("Simulating this game…"):
                    res = sim_game(g, day_str, min(ss["n_sims"], 3000))[0]
                game = dict(id=g["id"], date=day_str, start=g.get("startTimeUTC", ""))
            kind = st.segmented_control("Bet type", ["Single bet", "Same-game parlay"], default="Single bet",
                                        key="bp_kind") or "Single bet"
            if kind == "Single bet":
                bet_logger(res, game, "bp")
            else:
                parlay_builder(res, game, "bp")
        except Exception as e:
            st.error(f"Couldn't simulate that game ({e}).")

    with t_list:
        if bets.empty:
            st.info("No bets logged yet. Use ➕ Log a bet, or tick bets in 🔎 Find value.")
        else:
            if len(openb):
                st.markdown("**Open**")
                ov = openb.assign(Remove=False)[["Remove", "date", "game", "selection", "odds", "stake", "model_p", "book"]]
                ov = ov.rename(columns={"date": "Date", "game": "Game", "selection": "Bet", "odds": "Odds", "stake": "Stake",
                                        "model_p": "Model", "book": "Book"})
                ed = st.data_editor(ov, hide_index=True, width="stretch", key="open_ed",
                                    disabled=["Date", "Game", "Bet", "Odds", "Stake", "Model", "Book"],
                                    column_config={"Remove": st.column_config.CheckboxColumn(width="small"),
                                                   "Odds": st.column_config.NumberColumn(format="%+d"),
                                                   "Stake": st.column_config.NumberColumn(format="$%.2f"),
                                                   "Model": st.column_config.NumberColumn(format="percent")})
                gone = ed.index[ed["Remove"]]
                if len(gone) and st.button(f"🗑️ Remove {len(gone)} open bet(s)", key="rm_bets"):
                    save_store("bets", bets.drop(index=gone), f"Remove {len(gone)} bet(s)")
                    st.rerun()
            if len(done) or (bets["status"] == "Void").any():
                st.markdown("**Settled**")
                sv = bets[bets["status"] != "Open"].sort_values("graded_at", ascending=False)
                sv = pd.DataFrame({"": sv["status"].map({"Won": "✅", "Lost": "❌", "Push": "➖", "Void": "↩️"}),
                                   "Date": sv["date"], "Game": sv["game"], "Bet": sv["selection"], "Odds": sv["odds"],
                                   "Stake": sv["stake"], "Result": sv["result"], "Profit": sv["profit"]})
                st.dataframe(sv, hide_index=True, width="stretch", height=420, column_config={
                    "Odds": st.column_config.NumberColumn(format="%+d"),
                    "Stake": st.column_config.NumberColumn(format="$%.2f"),
                    "Profit": st.column_config.NumberColumn(format="$%+.2f")})
            st.download_button("⬇️ Download my bets (CSV)", bets.to_csv(index=False), "whl_my_bets.csv", "text/csv")

    with t_res:
        if done.empty:
            st.info("Results appear here once your bets are graded.")
        else:
            d = done.sort_values(["graded_at", "placed_at"]).reset_index(drop=True)
            fig = go.Figure(go.Scatter(x=list(range(1, len(d) + 1)), y=d["profit"].cumsum(), mode="lines+markers",
                                       line=dict(width=3, color="#27ae60" if profit >= 0 else "#c0392b"),
                                       hovertext=d["game"] + " · " + d["selection"], hoverinfo="text+y"))
            fig.add_hline(y=0, line=dict(color="#8ea0c4", dash="dot"))
            fig.update_layout(xaxis_title="Bet number", yaxis_title="Running profit ($)", margin=dict(t=10))
            chart(fig, "my_profit", 300)
            d["Market"] = d.apply(market_group, axis=1)
            grp = d.groupby("Market").agg(Bets=("status", "size"), Won=("status", lambda s: (s == "Won").sum()),
                                          Lost=("status", lambda s: (s == "Lost").sum()), Staked=("stake", "sum"),
                                          Profit=("profit", "sum")).reset_index()
            grp["ROI"] = 100 * grp["Profit"] / grp["Staked"].replace(0, np.nan)
            st.markdown("**By market**")
            st.dataframe(grp, hide_index=True, width="stretch", column_config={
                "Staked": st.column_config.NumberColumn(format="$%.0f"),
                "Profit": st.column_config.NumberColumn(format="$%+.2f"),
                "ROI": st.column_config.NumberColumn(format="%+.1f%%")})
            dm = d[d["model_p"].notna()].copy()
            if len(dm):
                dm["edge"] = dm["model_p"] - dm["odds"].apply(implied)
                dm["Model view"] = pd.cut(dm["edge"], [-1, -0.03, 0.0, 0.03, 1],
                                          labels=["Model disagreed (edge < -3%)", "Slightly against (-3–0%)",
                                                  "Slightly for (0–3%)", "Model agreed (edge > 3%)"])
                g2 = dm.groupby("Model view", observed=True).agg(
                    Bets=("status", "size"), Won=("status", lambda s: (s == "Won").mean() * 100),
                    Profit=("profit", "sum"), Staked=("stake", "sum")).reset_index()
                g2["ROI"] = 100 * g2["Profit"] / g2["Staked"].replace(0, np.nan)
                st.markdown("**Do bets the model liked do better?** (edge = model chance − book's implied chance)")
                st.dataframe(g2.drop(columns=["Staked"]), hide_index=True, width="stretch", column_config={
                    "Won": st.column_config.NumberColumn("Won %", format="%.0f%%"),
                    "Profit": st.column_config.NumberColumn(format="$%+.2f"),
                    "ROI": st.column_config.NumberColumn(format="%+.1f%%")})
            if len(done) < 50:
                st.caption(f"{len(done)} settled bets so far. Results swing a lot until you have a few hundred.")

    with t_edge:
        key = ""
        day_str = ss["day"].isoformat()
        games = day_games_safe(day_str)
        if not key:
            pass
        elif not games:
            st.info("No games on the selected date (change it on 📅 Games).")
        else:
            c1, c2 = st.columns(2)
            if c1.button(f"Scan game lines for {len(games)} games (3 credits)", key="scan_lines", width="stretch"):
                ss["_scan_lines_day"] = day_str
            do_props = c2.button(f"Scan player props too (~6 credits per game)", key="scan_props", width="stretch")
            if do_props:
                ss["_scan_props_day"] = day_str
            if ss.get("_scan_lines_day") == day_str or ss.get("_scan_props_day") == day_str:
                try:
                    feed, left = odds_feed(key)
                except Exception as e:
                    feed, left = [], "?"
                    st.error(f"Couldn't get odds ({e}).")
                rows, prow = [], []
                bar = st.progress(0.0, text="Simulating and comparing…")
                for k, g in enumerate(games):
                    h, a = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
                    evt = find_event(feed, h, a)
                    if evt and not _started(g.get("startTimeUTC", "")):
                        try:
                            res = sim_game(g, day_str, min(ss["n_sims"], 3000))[0]
                            game = dict(id=g["id"], date=day_str, start=g.get("startTimeUTC", ""))
                            rows += game_line_rows(res, evt, game)
                            if ss.get("_scan_props_day") == day_str:
                                prow += prop_rows(res, prop_feed(key, evt["id"])[0], game)
                        except Exception as e:
                            st.caption(f"{a} @ {h}: {e}")
                    bar.progress((k + 1) / len(games))
                bar.empty()
                min_ev = st.select_slider("Show bets with expected profit of at least", [0, 2, 5, 10, 20], value=2,
                                          format_func=lambda v: f"${v} per $100", key="edge_min")
                rows = [r for r in rows if r["EV/$100"] >= min_ev]
                prow = [r for r in prow if r["EV/$100"] >= min_ev]
                st.markdown(f"**Game lines** · {len(rows)} value bet(s)")
                edge_table(sorted(rows, key=lambda r: -r["EV/$100"]), "bp_lines", None)
                if ss.get("_scan_props_day") == day_str:
                    st.markdown(f"**Player props** · {len(prow)} value bet(s)")
                    edge_table(sorted(prow, key=lambda r: -r["EV/$100"])[:150], "bp_props", None)
                st.caption(f"API credits left this month: {left}. Started games are skipped.")
    st.caption("This is a model, not a guarantee. Bet only what you can afford to lose. "
               "If gambling stops being fun, call 1-800-GAMBLER.")


# ==========================================================================
# PAGE: Tracker (how good are the predictions?)
# ==========================================================================
def calib_chart(p, hit, key, title, bins=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01)):
    df = pd.DataFrame({"p": p, "hit": hit}).dropna()
    if df.empty:
        return
    df["bin"] = pd.cut(df["p"], list(bins), right=False)
    g = df.groupby("bin", observed=True).agg(n=("hit", "size"), pred=("p", "mean"), act=("hit", "mean")).reset_index()
    g = g[g["n"] > 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color="#aaa", dash="dot"), name="Perfect"))
    fig.add_trace(go.Scatter(x=g["pred"] * 100, y=g["act"] * 100, mode="markers+lines", name="Actual",
                             marker=dict(size=np.clip(np.sqrt(g["n"]) * 3, 7, 26), color="#2e86de"),
                             text=[f"{n} predictions" for n in g["n"]], hovertemplate="said %{x:.0f}% → happened %{y:.0f}%<br>%{text}<extra></extra>"))
    fig.update_layout(title=title, xaxis=dict(title="Model said (%)", range=[0, 100]),
                      yaxis=dict(title="Actually happened (%)", range=[0, 100]), margin=dict(t=40), showlegend=False)
    chart(fig, key, 300)


def page_tracker():
    st.markdown("## 📈 Tracker")
    st.caption("Every saved prediction is graded automatically after the game, so you can see whether the model's "
               "game AND player numbers hold up. Predictions lock when the game starts.")
    conf = _gh_conf()
    st.caption(f"💾 Saving to GitHub: **{conf['repo']}** · branch `{conf['branch']}`" if conf
               else f"💾 Saving to files next to the app: `{APP_DIR}`")
    today = local_today().isoformat()
    if st.button("🎲 Simulate & save today's games", type="primary", key="track_today",
                 help="Every game today with confirmed goalies; saves game and player predictions."):
        try:
            g_today = day_games_safe(today)
            if g_today:
                saved, saved_p = simulate_slate(today, g_today, min(ss["n_sims"], 3000), True, st.progress(0.0))
                st.success(f"Saved {saved} game and {saved_p} player prediction set(s)." if saved or saved_p
                           else "Nothing new to save (today's games have already started).")
            else:
                st.info("No WHL games today.")
        except Exception as e:
            st.error(f"Couldn't run today's games ({e}).")
    t_g, t_p, t_m, t_bt = st.tabs(["🏒 Games", "⛸️ Players", "🎯 Picks record", "🧪 Test on past games"])
    try:
        tdf, newly = grade_tracker()
        if newly:
            st.toast(f"Graded {newly} finished game(s).")
    except Exception as e:
        tdf = None
        st.error(f"Couldn't read or update the game tracker ({e}). If you set up GitHub, check the token and repo name.")
    done = None
    with t_g:
        if tdf is None or tdf.empty:
            st.info("No predictions saved yet. Use the button above, 🎲 on 📅 Games, or 📌 in the Game Center.")
        else:
            done = tdf[tdf["final_home"].notna()].copy()
            if len(done):
                home_won = (done["final_home"] > done["final_away"]).astype(float)
                pick_home = done["p_home"] >= 0.5
                correct = (home_won == 1) == pick_home
                brier = float(((done["p_home"] - home_won) ** 2).mean())
                tot_err = float((done["proj_home"] + done["proj_away"] - done["final_home"] - done["final_away"]).abs().mean())
                kpis([("Games graded", f"{len(done)}", ""),
                      ("Picks correct", f"{100 * correct.mean():.1f}%", f"{int(correct.sum())}-{int((~correct).sum())}"),
                      ("Brier score", f"{brier:.3f}", "lower is better · coin flip 0.250"),
                      ("Total goals miss", f"{tot_err:.2f}", "avg goals off per game")])
                p_pick = np.where(pick_home, done["p_home"], 1 - done["p_home"])
                calib_chart(p_pick, correct.astype(float).to_numpy(), "cal_games", "Are the win chances honest?",
                            bins=(0.5, 0.55, 0.6, 0.65, 0.7, 1.01))
                if len(done) < 50:
                    st.caption(f"Only {len(done)} games so far. These numbers bounce around until you have a few hundred.")
            else:
                st.info("No finished games yet. They're graded automatically the next time you open this page.")

            def _row(r):
                pick_ = r["home"] if r["p_home"] >= 0.5 else r["away"]
                pp_ = max(r["p_home"], 1 - r["p_home"]) * 100
                if pd.notna(r["final_home"]):
                    won = (r["final_home"] > r["final_away"]) == (r["p_home"] >= 0.5)
                    final = f"{r['away']} {int(r['final_away'])} – {int(r['final_home'])} {r['home']}" + \
                        (f" ({r['end']})" if r["end"] in ("OT", "SO") else "")
                    result = "✅" if won else "❌"
                else:
                    final, result = "", "⏳"
                return {"": result, "Date": r["date"], "Game": f"{r['away']} @ {r['home']}", "Pick": f"{pick_} {pp_:.0f}%",
                        "Projected": f"{r['proj_away']:.1f} – {r['proj_home']:.1f}", "Final": final,
                        "Goalies": f"{r['away_goalie']} vs {r['home_goalie']}"}

            st.markdown("**All saved game predictions**")
            st.dataframe(pd.DataFrame([_row(r) for _, r in tdf.sort_values(["date", "start_utc"], ascending=False).iterrows()]),
                         hide_index=True, width="stretch", height=380)
            st.download_button("⬇️ Download game predictions (CSV)", tdf.to_csv(index=False), "whl_predictions.csv", "text/csv")

    with t_p:
        try:
            pdf, pn = grade_players()
            if pn:
                st.toast(f"Graded {pn} player prediction(s).")
        except Exception as e:
            pdf = None
            st.error(f"Couldn't read or update the player tracker ({e}).")
        if pdf is None or pdf.empty:
            st.info("No player predictions saved yet. They're saved together with game predictions.")
        else:
            g = pdf[(pdf["status"] == "graded") & (pdf["pos"] != "G")].copy()
            gk = pdf[(pdf["status"] == "graded") & (pdf["pos"] == "G")].copy()
            if len(g):
                g["act_pts"] = g["act_g"] + g["act_a"]
                pt_hit = (g["act_pts"] > 0).astype(float)
                kpis([("Player-games graded", f"{len(g)}", f"{g['game_id'].nunique()} games"),
                      ("Shots: avg miss", f"{(g['proj_sog'] - g['act_sog']).abs().mean():.2f}",
                       f"bias {(g['proj_sog'] - g['act_sog']).mean():+.2f} per player"),
                      ("Points: avg miss", f"{(g['proj_pts'] - g['act_pts']).abs().mean():.2f}",
                       f"bias {(g['proj_pts'] - g['act_pts']).mean():+.2f}"),
                      ("1+ point Brier", f"{((g['p_point'] - pt_hit) ** 2).mean():.3f}",
                       f"baseline {pt_hit.mean() * (1 - pt_hit.mean()):.3f}"),
                      ("Goalie saves miss", f"{(gk['proj_saves'] - gk['act_saves']).abs().mean():.1f}" if len(gk) else "–",
                       f"{len(gk)} goalie games")])
                which = st.segmented_control("Check", ["Anytime goal", "1+ point", "3+ shots", "2+ points"],
                                             default="1+ point", key="trk_which") or "1+ point"
                col, hit = {"Anytime goal": ("p_goal", g["act_g"] > 0), "1+ point": ("p_point", g["act_pts"] > 0),
                            "3+ shots": ("p_sog3", g["act_sog"] >= 3), "2+ points": ("p_2pts", g["act_pts"] >= 2)}[which]
                calib_chart(g[col], hit.astype(float), "cal_players", f"{which}: are the chances honest?")
                g["miss"] = g["act_pts"] - g["proj_pts"]
                st.markdown("**Recent player results vs projections**")
                view = g.sort_values(["date", "game_id"], ascending=False).head(300)
                st.dataframe(pd.DataFrame({
                    "Date": view["date"], "Player": view["player"], "Team": view["team"], "Opp": view["opp"],
                    "Proj SOG": view["proj_sog"], "SOG": view["act_sog"], "Proj PTS": view["proj_pts"],
                    "G": view["act_g"], "A": view["act_a"], "Goal %": view["p_goal"] * 100, "Point %": view["p_point"] * 100}),
                    hide_index=True, width="stretch", height=380, column_config={
                        "Proj SOG": st.column_config.NumberColumn(format="%.1f"),
                        "Proj PTS": st.column_config.NumberColumn(format="%.2f"),
                        "Goal %": st.column_config.NumberColumn(format="%.0f%%"),
                        "Point %": st.column_config.NumberColumn(format="%.0f%%")})
                if g["game_id"].nunique() < 30:
                    st.caption("Player calibration needs a few hundred player-games before it means much.")
            else:
                st.info(f"{len(pdf)} player predictions saved; none graded yet. They're graded after each game ends.")
            st.download_button("⬇️ Download player predictions (CSV)", pdf.to_csv(index=False),
                               "whl_player_predictions.csv", "text/csv")

    with t_m:
        picks_record_ui()
        if done is None or done.empty:
            st.info("No graded games yet.")
        elif not (done["ml_home"].notna().any() or done["over_price"].notna().any()):
            st.info("Sportsbook odds aren't available for WHL games, so there are no saved prices to paper-bet. "
                    "Log the bets you actually make on the 💰 My Bets page to track them.")
        else:
            min_ev = st.select_slider("Only count bets with expected profit of at least (per $100)", [0, 2, 5, 10, 15],
                                      value=0, format_func=lambda v: f"${v}", key="paper_min")
            pb = bet_results(done, min_ev)
            if pb.empty:
                st.info("No bets met that bar.")
            else:
                wins, losses = int((pb["profit"] > 0).sum()), int((pb["profit"] < 0).sum())
                pr = float(pb["profit"].sum())
                kpis([("Bets", f"{len(pb)}", ""), ("Record", f"{wins}-{losses}", ""), ("Profit", money(pr), "$100 each"),
                      ("Return per bet", f"{pr / len(pb):+.1f}%", "")])
                curve = pb.sort_values("date").reset_index(drop=True)
                fig = go.Figure(go.Scatter(x=list(range(1, len(curve) + 1)), y=curve["profit"].cumsum(), mode="lines",
                                           line=dict(width=3, color="#27ae60" if pr >= 0 else "#c0392b"),
                                           hovertext=curve["game"] + " · " + curve["bet"], hoverinfo="text+y"))
                fig.add_hline(y=0, line=dict(color="#8ea0c4", dash="dot"))
                fig.update_layout(xaxis_title="Bet number", yaxis_title="Running profit ($)", margin=dict(t=10))
                chart(fig, "paper_curve", 280)

    with t_bt:
        backtest_ui()

    with st.expander("☁️ Keep your data on Streamlit Cloud (one-time GitHub setup)"):
        st.markdown('''
Streamlit Cloud wipes saved files whenever the app restarts, so on the cloud the tracker and your bets
are saved to your GitHub repo instead. On your PC you don't need this: they save to CSV files next to `whl_app.py`.

1. On github.com: your profile picture → **Settings** → **Developer settings** → **Personal access tokens**
   → **Fine-grained tokens** → **Generate new token**.
2. Name it `whl tracker`, set **Repository access** to *Only select repositories* → your `whl-sim` repo.
3. Under **Permissions → Repository permissions**, set **Contents** to **Read and write**. Generate and copy it.
4. In Streamlit Cloud: your app → **⋮** → **Settings** → **Secrets**, add (with your GitHub username), then **Save**:
```
GITHUB_TOKEN = "github_pat_paste_your_token_here"
GITHUB_REPO = "your-github-username/whl-sim"
```
Everything is saved on a separate branch called `tracker-data`, so saving never restarts your app.
''')


def backtest_ui():
    st.markdown("**How would the model have done on games already played?**")
    st.caption("Replays every finished game this season the way the model would have seen it that morning: each "
               "team's style comes only from its earlier games, player and goalie numbers are rebuilt from earlier "
               "games only, and the goalie who actually started is used. Nothing from later games leaks in. "
               "A team needs 5 earlier games before its games are tested.")
    if st.button("🧪 Run the test", type="primary", key="bt_run"):
        ss["bt_on"] = True
    if not ss.get("bt_on"):
        st.info("Takes about a minute the first time (it downloads every game this season), then it's cached.")
        return
    with st.spinner("Replaying this season's games…"):
        try:
            df = backtest_games(ss["n_recent"])
        except Exception as e:
            st.error(f"Couldn't run the test ({e}).")
            return
    if df.empty:
        st.info("Not enough finished games yet (each team needs 5).")
        return
    df = df.copy()
    pick_home = df["p_home"] >= 0.5
    correct = pick_home == (df["home_won"] == 1)
    brier = float(((df["p_home"] - df["home_won"]) ** 2).mean())
    base_p = float(df["home_won"].mean())
    base_brier = float(((base_p - df["home_won"]) ** 2).mean())
    conf = df[(df["p_home"] >= 0.6) | (df["p_home"] <= 0.4)]
    conf_ok = ((conf["p_home"] >= 0.5) == (conf["home_won"] == 1))
    tot_mae = float((df["proj_total"] - df["total"]).abs().mean())
    kpis([("Games tested", f"{len(df)}", f"{df['date'].min()} → {df['date'].max()}"),
          ("Winner picked", f"{100 * correct.mean():.1f}%", f"{int(correct.sum())}-{int((~correct).sum())}",
           "up" if correct.mean() > 0.55 else ""),
          ("Brier score", f"{brier:.3f}", f"always-home guess: {base_brier:.3f}", "up" if brier < base_brier else "down"),
          ("Confident picks (60%+)", f"{100 * conf_ok.mean():.0f}%" if len(conf) else "–", f"{len(conf)} games"),
          ("Total goals miss", f"{tot_mae:.2f}", f"avg {df['proj_total'].mean():.2f} proj vs {df['total'].mean():.2f} real")])
    c1, c2 = st.columns(2)
    with c1:
        p_pick = np.where(pick_home, df["p_home"], 1 - df["p_home"])
        calib_chart(p_pick, correct.astype(float).to_numpy(), "bt_cal", "Are the win chances honest?",
                    bins=(0.5, 0.55, 0.6, 0.65, 0.7, 1.01))
    with c2:
        dd = df.sort_values("date").reset_index(drop=True)
        dd["acc"] = (((dd["p_home"] >= 0.5) == (dd["home_won"] == 1)).astype(float)).expanding().mean() * 100
        fig = go.Figure(go.Scatter(x=list(range(1, len(dd) + 1)), y=dd["acc"], mode="lines",
                                   line=dict(width=3, color="#00d26a")))
        fig.add_hline(y=50, line=dict(color="#8ea0c4", dash="dot"))
        fig.update_layout(title="Winner-picking accuracy as the season went on", xaxis_title="Games tested",
                          yaxis_title="% correct", margin=dict(t=40), yaxis=dict(range=[30, 80]))
        chart(fig, "bt_acc", 300)
    over = (df["total"] > 6.5).astype(float)
    calib_chart(df["p_over65"], over.to_numpy(), "bt_over", "Over 6.5 goals: are the chances honest?")
    st.markdown("**Every tested game**")
    view = df.sort_values("date", ascending=False)
    st.dataframe(pd.DataFrame({
        "": np.where(((view["p_home"] >= 0.5) == (view["home_won"] == 1)), "✅", "❌"), "Date": view["date"],
        "Game": view["away"] + " @ " + view["home"],
        "Pick": np.where(view["p_home"] >= 0.5, view["home"], view["away"]) + " " +
                (np.maximum(view["p_home"], 1 - view["p_home"]) * 100).round(0).astype(int).astype(str) + "%",
        "Final": view["away"] + " " + view["as_"].astype(str) + " – " + view["hs"].astype(str) + " " + view["home"],
        "Proj total": view["proj_total"], "Real total": view["total"]}),
        hide_index=True, width="stretch", height=360,
        column_config={"Proj total": st.column_config.NumberColumn(format="%.1f")})
    st.caption("This can't show betting profit: past sportsbook odds aren't available on the free odds plan. "
               "It shows whether the model's chances match what happened, which is what makes the edges trustworthy.")

    st.markdown("**Player props on the most recent games**")
    k = st.select_slider("Games to test (full simulation of each)", [25, 50, 100, 200], value=50, key="bt_np")
    if st.button(f"⛸️ Test player props on the last {k} games", key="bt_players"):
        ss["bt_players_k"] = k
    kk = ss.get("bt_players_k")
    if kk:
        ids = df.sort_values("date")["game_id"].tolist()[-kk:]
        rows = []
        bar = st.progress(0.0, text="Simulating player stats…")
        for j, gid in enumerate(ids):
            try:
                rows += backtest_players(int(gid), ss["n_recent"])
            except Exception:
                pass
            bar.progress((j + 1) / len(ids))
        bar.empty()
        pdf = pd.DataFrame(rows)
        if pdf.empty:
            st.info("No player results.")
            return
        hit = ((pdf["act_g"] + pdf["act_a"]) > 0).astype(float)
        b_model = float(((pdf["p_point"] - hit) ** 2).mean())
        b_base = float(((pdf["base_point"] - hit) ** 2).mean())
        kpis([("Player-games", f"{len(pdf)}", f"{pdf['game_id'].nunique()} games"),
              ("1+ point Brier", f"{b_model:.3f}", f"season-average guess: {b_base:.3f}", "up" if b_model < b_base else "down"),
              ("Shots: avg miss", f"{(pdf['proj_sog'] - pdf['act_sog']).abs().mean():.2f}",
               f"bias {(pdf['proj_sog'] - pdf['act_sog']).mean():+.2f}"),
              ("Points: avg miss", f"{(pdf['proj_pts'] - pdf['act_g'] - pdf['act_a']).abs().mean():.2f}", "")])
        which = st.segmented_control("Check", ["1+ point", "Anytime goal", "3+ shots"], default="1+ point",
                                     key="bt_which") or "1+ point"
        col, h2 = {"1+ point": ("p_point", hit), "Anytime goal": ("p_goal", (pdf["act_g"] > 0).astype(float)),
                   "3+ shots": ("p_sog3", (pdf["act_sog"] >= 3).astype(float))}[which]
        calib_chart(pdf[col], h2.to_numpy(), "bt_pcal", f"{which}: are the chances honest?")
        st.caption("Uses the players who actually dressed and the starting goalie, like a lineup confirmed at warmups.")


# ==========================================================================
# PAGE: Season & playoffs
# ==========================================================================
def league_ready(key):
    if st.session_state.get("league_ready") == ss["n_recent"]:
        return True
    st.info("The season and playoff simulators use every WHL team's real game sheets "
            f"(about {16 * ss['n_recent']} games). The first build takes 1–2 minutes, then it's cached for hours.")
    if st.button("Build the league model", type="primary", key=key):
        with st.spinner("Downloading game sheets for every WHL team and simulating every matchup…"):
            league_model(ss["n_recent"])
        st.session_state["league_ready"] = ss["n_recent"]
        st.rerun()
    return False


PCT_COLS = ["Playoffs %", "Division %", "Best record %", "Conf final %", "Final %", "Cup %"]


def pct_config(cols):
    return {c: st.column_config.ProgressColumn(c, min_value=0, max_value=100, format="%.1f%%") for c in cols}


def page_season():
    top = st.columns([3, 2], vertical_alignment="bottom")
    top[0].markdown("## 🏆 Season & playoffs")
    with top[1]:
        settings_popover()
    t_season, t_series, t_bracket = st.tabs(["📅 Rest of season", "⚔️ Series", "🥅 Bracket today"])
    n_recent = ss["n_recent"]
    with t_season:
        st.caption("Plays every remaining regular-season game thousands of times, adds the results to the real "
                   "standings, then plays out the WHL playoffs each time (top 8 per conference, division winners "
                   "seeded 1–2, re-seeded after the first round).")
        if league_ready("league_btn_season"):
            n_seasons = st.select_slider("Seasons to simulate", [1000, 2000, 5000, 10000], value=2000, key="n_seasons")
            with st.spinner(f"Simulating {n_seasons:,} seasons…"):
                sres = season_sim(n_recent, n_seasons)
            tbl = sres["table"]
            if not ss.get(f"_odds_saved_{local_today().isoformat()}"):
                ss[f"_odds_saved_{local_today().isoformat()}"] = True
                try:
                    save_title_odds(tbl)
                except Exception:
                    pass
            st.caption(f"{sres['n_games']} games left · {sres['n_seasons']:,} simulated seasons")
            fmt_cols = {"Proj PTS": st.column_config.NumberColumn(format="%.1f"),
                        "Low (10%)": st.column_config.NumberColumn(format="%.0f"),
                        "High (90%)": st.column_config.NumberColumn(format="%.0f")}
            view = st.segmented_control("View", ["By division", "Whole league"], default="By division", key="season_view")
            if view == "Whole league":
                st.dataframe(tbl, hide_index=True, width="stretch", height=1150,
                             column_config={**pct_config(PCT_COLS), **fmt_cols})
            else:
                for conf, ds in CONF_DIVS.items():
                    st.markdown(f"#### {conf} Conference")
                    cols = st.columns(2)
                    for col, d in zip(cols, ds):
                        with col:
                            st.markdown(f"**{d}**")
                            sub = tbl[tbl["Div"] == d][["Team", "PTS now", "Proj PTS", "Playoffs %", "Cup %"]]
                            st.dataframe(sub, hide_index=True, width="stretch", column_config={
                                **{c: st.column_config.NumberColumn(format="%.1f%%") for c in PCT_COLS}, **fmt_cols})
            topc = tbl.sort_values("Cup %", ascending=False).head(16)
            fig = go.Figure(go.Bar(x=topc["Team"], y=topc["Cup %"], marker_color=[TEAM_COLORS.get(t, "#888") for t in topc["Team"]],
                                   text=[f"{v:.1f}%" for v in topc["Cup %"]], textposition="outside", cliponaxis=False))
            fig.update_layout(title="Ed Chynoweth Cup (WHL championship) odds", yaxis_title="%", margin=dict(t=40))
            chart(fig, "cup_chart", 340)
    with t_series:
        st.caption("Best-of-7, 2-2-1-1-1 format, playoff overtime (sudden-death 5-on-5).")
        c1, c2 = st.columns(2)
        sa_ = c1.selectbox("Team A", TEAMS, index=TEAMS.index(ss["home"]), format_func=_label, key="ser_a")
        sb_ = c2.selectbox("Team B", TEAMS, index=TEAMS.index(ss["away"]), format_func=_label, key="ser_b")
        if sa_ == sb_:
            st.warning("Pick two different teams.")
        else:
            hi_choice = st.radio("Home ice", [sa_, sb_], horizontal=True, key="ser_hi")
            with st.spinner("Simulating the series…"):
                a_home, a_road = pair_series(sa_, sb_, n_recent)
            if hi_choice == sa_:
                hi_t, lo_t, ph, pa = sa_, sb_, a_home, a_road
            else:
                hi_t, lo_t, ph, pa = sb_, sa_, 1 - a_road, 1 - a_home
            o = series_outcomes(ph, pa)
            p_hi = sum(v for (w, _), v in o.items() if w == "hi")
            kpis([(f"{hi_t} wins series", pct1(p_hi), ""), (f"{lo_t} wins series", pct1(1 - p_hi), ""),
                  (f"{hi_t} per game at home", pct1(ph), ""), (f"{hi_t} per game on road", pct1(pa), "")])
            ch2, ca2 = team_colors(hi_t, lo_t)
            labels = [f"{hi_t} in {g}" for g in (4, 5, 6, 7)] + [f"{lo_t} in {g}" for g in (4, 5, 6, 7)]
            vals = [o[("hi", g)] * 100 for g in (4, 5, 6, 7)] + [o[("lo", g)] * 100 for g in (4, 5, 6, 7)]
            fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=[ch2] * 4 + [ca2] * 4,
                                   text=[f"{v:.1f}%" for v in vals], textposition="outside", cliponaxis=False))
            fig.update_layout(yaxis_title="% of series", margin=dict(t=20))
            chart(fig, "series_chart", 320)
    with t_bracket:
        if league_ready("league_btn_playoffs"):
            bt = today_bracket(n_recent)
            teams_l, SP, key_now, br, counts, n_br = bt["teams"], bt["SP"], bt["key"], bt["br"], bt["counts"], bt["n"]
            if not br:
                st.info("Not enough standings data to build a bracket yet.")
            else:
                seed_of = {t: i + 1 for seeds in br.values() for i, t in enumerate(seeds)}
                cols = st.columns(2)
                for col, (conf, seeds) in zip(cols, br.items()):
                    with col:
                        st.markdown(f"#### {conf} — first round")
                        for i in range(4):
                            a_, b_ = seeds[i], seeds[7 - i]
                            hi, lo = (a_, b_) if key_now[a_] >= key_now[b_] else (b_, a_)
                            p = SP[hi, lo]
                            st.write(f"({seed_of[hi]}) **{teams_l[hi]}** {100 * p:.0f}%  vs  "
                                     f"({seed_of[lo]}) **{teams_l[lo]}** {100 * (1 - p):.0f}%")
                st.caption("Division winners are seeded 1–2; the bracket is re-seeded after the first round.")
                playoff_teams = [i for i in range(len(teams_l)) if counts[i, 0] > 0]
                adv = pd.DataFrame({
                    "Seed": [seed_of.get(i, "") for i in playoff_teams], "Team": [teams_l[i] for i in playoff_teams],
                    "Round 2 %": counts[playoff_teams, 1] / n_br * 100, "Conf final %": counts[playoff_teams, 2] / n_br * 100,
                    "Final %": counts[playoff_teams, 3] / n_br * 100, "Champion %": counts[playoff_teams, 4] / n_br * 100,
                }).sort_values("Champion %", ascending=False)
                st.dataframe(adv, hide_index=True, width="stretch", height=600,
                             column_config=pct_config(["Round 2 %", "Conf final %", "Final %", "Champion %"]))


# ==========================================================================
# PAGE: Teams
# ==========================================================================
def page_teams():
    top = st.columns([3, 2], vertical_alignment="bottom")
    top[0].markdown("## 📊 Teams")
    with top[1]:
        settings_popover()
    t_style, t_stand = st.tabs(["🧭 How a team plays", "📋 Standings"])
    with t_style:
        team = st.selectbox("Team", TEAMS, index=TEAMS.index(ss["home"]), format_func=_label, key="style_team")
        prof = build_profile(team, ss["n_recent"])
        color = TEAM_COLORS.get(team, "#2e86de")
        st.caption(f"Measured from {team}'s real game sheets in its last {prof['n_games']} games "
                   "(newer games count more). This is what drives the simulation.")
        kpis([("Shots for / game", f"{prof['sog_pg']:.1f}", ""), ("Shots against / game", f"{prof['soga_pg']:.1f}", ""),
              ("Goals for / against", f"{prof['gf_pg']:.2f} / {prof['ga_pg']:.2f}", ""),
              ("Power play", pct(prof["pp_pct"]), ""), ("Penalty kill", pct(prof["pk_pct"]), ""),
              ("Faceoffs", pct(prof["fo_pct"]), "")])
        st.caption("The WHL doesn't publish shot locations, so there are no shot maps; the simulator uses "
                   "league-typical shot locations.")
        lab = ["Leading", "Tied", "Trailing", "Power play", "Shorthanded"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=lab, y=[prof["r_for"][s] for s in STATES], name="Shots for /60", marker_color=color))
        fig.add_trace(go.Bar(x=lab, y=[prof["r_ag"][s] for s in STATES], name="Shots against /60", marker_color="#8ea0c4"))
        fig.add_trace(go.Scatter(x=lab, y=[LG_RATE[s] for s in STATES], mode="markers", name="League avg",
                                 marker=dict(symbol="line-ew-open", size=28, color="#ffd84d", line=dict(width=3))))
        fig.update_layout(barmode="group", title="Shot rates by game state (from real shot totals)", margin=dict(t=40),
                          legend=dict(orientation="h", y=-0.25))
        chart(fig, f"rates_{team}", 330)
        if prof["recent_games"]:
            st.markdown("**Recent games**")
            st.dataframe(pd.DataFrame(prof["recent_games"][::-1]).rename(
                columns={"date": "Date", "opp": "Opponent", "result": "Result"}), hide_index=True, width="stretch", height=260)
    with t_stand:
        tb = standings.copy()
        tb["GF/GP"] = tb["gf"] / tb["gp"].clip(lower=1)
        tb["GA/GP"] = tb["ga"] / tb["gp"].clip(lower=1)
        view = tb.rename(columns={"team": "Team", "gp": "GP", "w": "W", "l": "L", "otl": "OTL", "pts": "PTS",
                                  "l10": "L10", "streak": "Strk", "div": "Div"})
        cols = ["Team", "GP", "W", "L", "OTL", "PTS", "GF/GP", "GA/GP", "L10", "Strk"]
        cfg = {"GF/GP": st.column_config.NumberColumn(format="%.2f"), "GA/GP": st.column_config.NumberColumn(format="%.2f")}
        for conf, ds in CONF_DIVS.items():
            st.markdown(f"#### {conf}")
            cc = st.columns(2)
            for col, d in zip(cc, ds):
                with col:
                    st.markdown(f"**{d}**")
                    st.dataframe(view[view["Div"] == d].sort_values("PTS", ascending=False)[cols], hide_index=True,
                                 width="stretch", column_config=cfg)


# ==========================================================================
# WHL extras: picks of the day, what-if lab, Beat the Bot, races, rivalries
# ==========================================================================
FUN_CSS = """<style>
.pod { border-radius: 14px; padding: 10px 14px; background: rgba(128,128,128,.09); height: 100%; }
.pod .h { font-size: .78rem; font-weight: 800; letter-spacing: .6px; text-transform: uppercase; opacity: .8; }
.pod .t { font-size: 1.15rem; font-weight: 900; margin: 2px 0; display: flex; align-items: center; gap: 8px; }
.pod .t img { width: 30px; height: 30px; }
.pod .s { font-size: .8rem; opacity: .75; }
.pod.lock { box-shadow: inset 3px 0 0 #00d26a; } .pod.upset { box-shadow: inset 3px 0 0 #ff5c7a; }
.pod.flip { box-shadow: inset 3px 0 0 #ffb020; }
.tug { display: flex; height: 16px; border-radius: 8px; overflow: hidden; margin: 6px 0 2px; }
.tug div { display: flex; align-items: center; justify-content: center; font-size: .7rem; font-weight: 800; color: #fff;
           text-shadow: 0 1px 2px #000a; transition: width .5s; }
.verdict { border-radius: 12px; padding: 8px 12px; margin: 6px 0; font-weight: 700; }
.verdict.good { background: rgba(0,210,106,.14); } .verdict.ok { background: rgba(128,128,128,.12); }
.verdict.bad { background: rgba(229,72,77,.14); }
.pk { border-radius: 14px; padding: 8px 12px; background: rgba(128,128,128,.07); margin-bottom: 4px; }
.pk .row { display: flex; align-items: center; justify-content: space-between; gap: 8px; font-weight: 800; }
.pk .row span { display: flex; align-items: center; gap: 6px; } .pk img { width: 26px; height: 26px; }
.pk .s { font-size: .78rem; opacity: .75; }
.lb-me { font-weight: 800; }
.confetti { position: fixed; inset: 0; pointer-events: none; z-index: 999999; overflow: hidden; }
.confetti i { position: absolute; top: -14px; width: 9px; height: 14px; border-radius: 2px; opacity: 0;
              animation-name: cfall; animation-timing-function: cubic-bezier(.2,.6,.4,1); animation-fill-mode: forwards; }
@keyframes cfall { 0% { opacity: 1; transform: translateY(0) rotate(0deg); }
                   85% { opacity: 1; } 100% { opacity: 0; transform: translateY(104vh) rotate(760deg); } }
.trophy { text-align: center; font-size: 1.05rem; font-weight: 800; padding: 10px; border-radius: 14px;
          background: linear-gradient(90deg, rgba(255,216,77,.12), rgba(0,210,106,.12)); margin: 6px 0; }
</style>"""
st.markdown(FUN_CSS, unsafe_allow_html=True)


def confetti(n=90):
    r = np.random.default_rng()
    cols = ["#00d26a", "#4fc3ff", "#ffb020", "#ff5c7a", "#b388ff", "#ffd84d", "#ffffff"]
    spans = "".join(f'<i style="left:{r.uniform(0, 100):.1f}%;background:{cols[k % len(cols)]};'
                    f'animation-delay:{r.uniform(0, 0.9):.2f}s;animation-duration:{r.uniform(2.2, 3.8):.2f}s;'
                    f'width:{r.integers(6, 11)}px"></i>' for k in range(n))
    st.markdown(f'<div class="confetti">{spans}</div>', unsafe_allow_html=True)


def fatigue_text(fat, home, away):
    bits = [f"😴 {t}: {fat[k]['text']}" for t, k in ((away, "away"), (home, "home"))
            if fat.get(k) and fat[k].get("text") and not fat[k]["text"].startswith("rested")]
    return " · ".join(bits)


# ---------------- picks of the day ----------------
def day_predictions(day_str):
    """{game id: prediction} for a date: saved predictions (daily robot / Tracker) plus this session's sims."""
    preds = {}
    try:
        df = load_tracker()
        for _, r in df[df["date"] == day_str].iterrows():
            if pd.isna(r["p_home"]):
                continue
            preds[int(r["game_id"])] = dict(home=r["home"], away=r["away"], p_home=float(r["p_home"]),
                                            likely_ah=str(r["likely"] or ""), start=str(r["start_utc"] or ""),
                                            total=float((r["proj_away"] or 0) + (r["proj_home"] or 0)))
    except Exception:
        pass
    for gid, s in ss.get("slate", {}).get(day_str, {}).items():
        if "p_home" in s:
            preds[gid] = dict(s)
    return preds


def pts_pct(team):
    try:
        r = STN.loc[team]
        return float(r.pts) / max(2 * int(r.gp), 1)
    except Exception:
        return 0.5


@st.cache_data(ttl=600, show_spinner=False)
def model_record():
    """Season record of the model's picks: every game, and the 'lock of the night'."""
    df = load_tracker()
    g = df[df["final_home"].notna() & df["p_home"].notna()].copy()
    if g.empty:
        return None
    g["right"] = (g["p_home"] >= 0.5) == (g["final_home"] > g["final_away"])
    g["conf"] = (g["p_home"] - 0.5).abs()
    locks = g.loc[g.groupby("date")["conf"].idxmax()]
    return dict(w=int(g["right"].sum()), l=int((~g["right"]).sum()), lw=int(locks["right"].sum()),
                ll=int((~locks["right"]).sum()))


def picks_panel(day_str):
    preds = day_predictions(day_str)
    st.markdown("#### 🎯 Picks of the day")
    if not preds:
        st.caption("Tap 🎲 Simulate all games above (or let the daily robot run) to see the model's lock of the night, "
                   "an upset alert and the closest game.")
        return
    finals = final_scores(day_str)

    def result(gid, team):
        sc = finals.get(gid)
        if not sc or sc[0] is None:
            return ""
        p = preds[gid]
        winner = p["home"] if sc[1] > sc[0] else p["away"]
        return " ✅" if winner == team else " ❌"

    items = list(preds.items())
    lock_gid, lock = max(items, key=lambda kv: abs(kv[1]["p_home"] - 0.5))
    flip_gid, flip = min(items, key=lambda kv: abs(kv[1]["p_home"] - 0.5))
    lock_team = lock["home"] if lock["p_home"] >= 0.5 else lock["away"]
    lock_p = max(lock["p_home"], 1 - lock["p_home"])
    dogs = []
    skip = {lock_gid} | ({flip_gid} if len(items) >= 3 else set())
    for gid, p in items:
        if gid in skip:
            continue
        dog = p["home"] if pts_pct(p["home"]) < pts_pct(p["away"]) else p["away"]
        pd_ = p["p_home"] if dog == p["home"] else 1 - p["p_home"]
        if pd_ >= 0.40:
            dogs.append((pd_, gid, dog, p))
    cols = st.columns(3)

    def card(col, cls, head, team, line, sub):
        col.markdown(f'<div class="pod {cls}"><div class="h">{head}</div><div class="t"><img src="{logo_url(team)}" alt="">'
                     f'{esc(team)}{line}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)

    card(cols[0], "lock", "🔒 Lock of the night", lock_team, result(lock_gid, lock_team),
         f"{pct1(lock_p)} to win · {esc(lock['away'])} @ {esc(lock['home'])}")
    if dogs:
        pd_, gid, dog, p = max(dogs, key=lambda d: d[0])
        fav = p["away"] if dog == p["home"] else p["home"]
        card(cols[1], "upset", "🚨 Upset alert", dog, result(gid, dog),
             f"{pct1(pd_)} vs {esc(fav)}, who have the better record")
    else:
        cols[1].markdown('<div class="pod upset"><div class="h">🚨 Upset alert</div><div class="t">None tonight</div>'
                         '<div class="s">Every underdog is below 40%.</div></div>', unsafe_allow_html=True)
    ft = flip["home"] if flip["p_home"] >= 0.5 else flip["away"]
    card(cols[2], "flip", "⚖️ Coin flip of the night", ft, result(flip_gid, ft),
         f"{esc(flip['away'])} {100 * (1 - flip['p_home']):.0f}% – {100 * flip['p_home']:.0f}% {esc(flip['home'])}")
    rec = model_record()
    line = ""
    if rec:
        tot = rec["w"] + rec["l"]
        line = (f"Model this season: **{rec['w']}-{rec['l']}** ({100 * rec['w'] / max(tot, 1):.1f}%) · "
                f"locks **{rec['lw']}-{rec['ll']}**. ")
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    c1.markdown(line + "Think you can do better?")
    if c2.button("🎮 Beat the Bot", width="stretch", key="pod_play"):
        st.switch_page(PAGE["pickem"])


def picks_record_ui():
    """The model's straight-up record, its 'lock of the night' record and recent locks."""
    df = load_tracker()
    g = df[df["final_home"].notna() & df["p_home"].notna()].copy()
    if g.empty:
        st.info("No graded games yet.")
        return
    g["pick"] = np.where(g["p_home"] >= 0.5, g["home"], g["away"])
    g["conf"] = np.maximum(g["p_home"], 1 - g["p_home"])
    g["right"] = (g["p_home"] >= 0.5) == (g["final_home"] > g["final_away"])
    locks = g.loc[g.groupby("date")["conf"].idxmax()].sort_values("date", ascending=False)
    big = g[g["conf"] >= 0.65]
    kpis([("All picks", f"{int(g['right'].sum())}-{int((~g['right']).sum())}", pct1(g["right"].mean())),
          ("Locks of the night", f"{int(locks['right'].sum())}-{int((~locks['right']).sum())}", pct1(locks["right"].mean())),
          ("65%+ favourites", f"{int(big['right'].sum())}-{int((~big['right']).sum())}" if len(big) else "–",
           pct1(big["right"].mean()) if len(big) else ""),
          ("Toss-ups (under 55%)", pct1(g[g["conf"] < 0.55]["right"].mean()) if (g["conf"] < 0.55).any() else "–", "right")])
    st.markdown("**🔒 Recent locks of the night**")
    st.dataframe(pd.DataFrame({"Date": locks["date"], "Game": locks["away"] + " @ " + locks["home"], "Pick": locks["pick"],
                               "Chance": 100 * locks["conf"],
                               "Final": [f"{int(a)}–{int(b)}" for a, b in zip(locks["final_away"], locks["final_home"])],
                               "": np.where(locks["right"], "✅", "❌")}).head(20),
                 hide_index=True, width="stretch", column_config={"Chance": st.column_config.NumberColumn(format="%.0f%%")})
    st.divider()


# ---------------- what-if lab ----------------
def whatif_lab(mi, res):
    home, away = mi["home"], mi["away"]
    hp, ap = mi["hp"], mi["ap"]
    ng = mi["ng"]
    date = ng["date"] if ng else None
    playoff = bool(ss["playoff"] or (ng and ng.get("playoff")))
    fat = res.get("fatigue") or {}

    @st.fragment
    def lab():
        st.caption("Change something and the win chance updates right away (2,000 quick simulations each). "
                   "Compared with the same quick simulation of tonight's real setup.")
        cols = st.columns(2)
        picks = {}
        for col, side, prof, cur_g, cur_out in ((cols[0], "away", ap, mi["ag"], mi["ao"]),
                                                 (cols[1], "home", hp, mi["hg"], mi["ho"])):
            with col:
                st.markdown(f'**{_label(prof["team"])}**')
                ids = [g["id"] for g in prof["goalies"]]
                g = st.selectbox("🥅 Goalie", ids, index=ids.index(cur_g) if cur_g in ids else 0, key=f"wi_g_{side}",
                                 format_func=lambda i, p=prof: goalie_name(p, i)) if ids else None
                stars = sorted([s for s in prof["skaters"] if s["id"] not in cur_out], key=lambda s: -(s["g"] + s["a"]))[:8]
                sit = st.multiselect("🚑 Sit a star", [s["id"] for s in stars], key=f"wi_sit_{side}",
                                     format_func=lambda i, ss_=stars: next(f"{s['name']} ({s['g'] + s['a']} pts)" for s in ss_ if s["id"] == i))
                lvl0 = (fat.get(side) or {}).get("level", "Normal")
                lvl = st.select_slider("🔋 Energy", list(FATIGUE_LEVELS), value=lvl0, key=f"wi_f_{side}",
                                       help="Fresh = well rested · Tired = back-to-back or a long bus ride · "
                                            "Exhausted = third game in three nights after a long trip")
                picks[side] = (g, tuple(sorted(set(cur_out) | set(sit))), lvl)
        hi = st.toggle("🏟️ Home-ice advantage", value=True, key="wi_hi")
        base_lv = ((fat.get("home") or {}).get("level", "Normal"), (fat.get("away") or {}).get("level", "Normal"))
        base = run_matchup(home, away, mi["hg"], mi["ag"], mi["ho"], mi["ao"], 2000, playoff, ss["n_recent"], 7, date,
                           False, base_lv, True)
        new = run_matchup(home, away, picks["home"][0], picks["away"][0], picks["home"][1], picks["away"][1], 2000,
                          playoff, ss["n_recent"], 7, date, False, (picks["home"][2], picks["away"][2]), hi)
        d = round(new["home_win"] - base["home_win"], 3) + 0.0
        ch, ca = team_colors(home, away)
        st.markdown(f'<div class="tug"><div style="width:{100 * new["away_win"]:.1f}%;background:{ca}">{away} '
                    f'{100 * new["away_win"]:.0f}%</div><div style="width:{100 * new["home_win"]:.1f}%;background:{ch}">'
                    f'{home} {100 * new["home_win"]:.0f}%</div></div>', unsafe_allow_html=True)
        arrow = lambda v: "up" if v > 0.005 else ("down" if v < -0.005 else "")
        kpis([(f"{away} win", pct1(new["away_win"]), f"{-100 * d + 0.0:+.1f} pts vs tonight", arrow(-d)),
              (f"{home} win", pct1(new["home_win"]), f"{100 * d:+.1f} pts vs tonight", arrow(d)),
              ("Projected score", f"{new['goals_mean'][1]:.1f} – {new['goals_mean'][0]:.1f}",
               f"was {base['goals_mean'][1]:.1f} – {base['goals_mean'][0]:.1f}"),
              ("Total goals", f"{new['total'].mean():.2f}", f"was {base['total'].mean():.2f}"),
              ("Goes to OT/SO", pct1(new["ot"] + new["so"]), f"was {pct1(base['ot'] + base['so'])}")])
        if abs(d) >= 0.08:
            st.success(f"Big swing! That moves {home if d > 0 else away} up {abs(100 * d):.0f} points.")
        if st.button("↺ Reset the lab", key="wi_reset"):
            for k in list(ss.keys()):
                if str(k).startswith("wi_"):
                    del ss[k]
            st.rerun()

    lab()


# ---------------- Beat the Bot ----------------
def pickem_player():
    qp = st.query_params.get("player", "")
    if "pk_name" not in ss:
        ss["pk_name"] = qp
    return ss["pk_name"]


def _set_name():
    ss["pk_name"] = ss["_w_pk_name"].strip()[:24]
    if ss["pk_name"]:
        st.query_params["player"] = ss["pk_name"]


def page_pickem():
    top = st.columns([3, 2, 2], vertical_alignment="bottom")
    top[0].markdown("## 🎮 Beat the Bot")
    name = pickem_player()
    top[1].text_input("Your name", value=name, key="_w_pk_name", on_change=_set_name, max_chars=24,
                      placeholder="Type your name", label_visibility="collapsed")
    day = top[2].date_input("Date", value=ss["day"], key="_w_pk_day", label_visibility="collapsed")
    day_str = day.isoformat()
    name = ss.get("pk_name", "")
    code = secret("PICKEM_CODE", "")
    st.caption("Pick tonight's winners before puck drop, add a score guess for bonus points, and see if you can "
               "out-pick the simulator. **10** for each winner · **+5** for an upset the Bot gave under 40% · "
               "**+15** exact score · **+5** if both teams' goals are within one.")
    try:
        df, n_graded = grade_pickem()
    except Exception as e:
        st.warning(f"Couldn't check results right now ({e}).")
        df = load_store("pickem")
    t_pick, t_lb, t_mine, t_how = st.tabs(["🗳️ Make picks", "🏆 Leaderboard", "📜 My picks", "🏅 Badges"])

    with t_pick:
        games = day_games_safe(day_str)
        if not games:
            st.info(f"No WHL games on {day:%A, %B %d}. Pick another date.")
        else:
            preds = day_predictions(day_str)
            have = set(zip(df["player"], pd.to_numeric(df["game_id"], errors="coerce")))
            need = {gid: p for gid, p in preds.items() if (BOT, gid) not in have}
            if need and not ss.get(f"_bot_{day_str}"):
                ss[f"_bot_{day_str}"] = True
                try:
                    if save_picks(bot_rows(day_str, need), f"Bot picks for {day_str}"):
                        df = load_store("pickem")
                except Exception:
                    pass
            if not preds:
                st.caption("🤖 The Bot makes its picks once tonight's games are simulated (🎲 on the 📅 Games page, "
                           "or automatically by the daily robot).")
            if not name:
                st.info("👆 Type your name at the top to play. Friends can join with their own names, and your picks "
                        "are saved for the season leaderboard.")
            locked_ok = not code or ss.get("pk_code_ok")
            if name and not locked_ok:
                c = st.text_input("League code", type="password", key="_w_pk_code",
                                  help="The app owner set a code so only friends can add picks.")
                if c and c == code:
                    ss["pk_code_ok"] = True
                    st.rerun()
                elif c:
                    st.error("That code isn't right.")
            mine = df[df["player"] == name] if name else df.iloc[0:0]
            bot = df[df["player"] == BOT]
            prev = {int(r["game_id"]): r for _, r in mine.iterrows()}
            bots = {int(r["game_id"]): r for _, r in bot.iterrows()}
            with st.form(f"pk_form_{day_str}", border=False):
                rows, dropped = [], []
                open_n = 0
                for g in games:
                    h, a = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
                    gid = g["id"]
                    started = _started(g.get("startTimeUTC", "")) or g.get("gameState") in ("LIVE", "CRIT", "OFF", "FINAL")
                    pr, bt = prev.get(gid), bots.get(gid)
                    when = local_time(g.get("startTimeUTC")) or ""
                    with st.container(border=True):
                        st.markdown(f'<div class="pk"><div class="row"><span><img src="{logo_url(a)}" alt="">{a}</span>'
                                    f'<span class="s">{"🔒 locked" if started else "⏰ " + esc(when)}</span>'
                                    f'<span>{h}<img src="{logo_url(h)}" alt=""></span></div></div>', unsafe_allow_html=True)
                        if started:
                            bits = []
                            if pr is not None:
                                sc = (f" ({int(pr['score_away'])}–{int(pr['score_home'])})"
                                      if pd.notna(pr["score_away"]) else "")
                                res_ = (f" · {pr['note']} · **{int(pr['points'])} pts**" if pr["status"] in ("Won", "Lost") else "")
                                bits.append(f"You: **{pr['pick']}**{sc}{res_}")
                            if bt is not None:
                                bits.append(f"🤖 Bot: **{bt['pick']}**" + (f" · {int(bt['points'])} pts" if bt["status"] in ("Won", "Lost") else ""))
                            st.caption(" · ".join(bits) or "Game has started – picks are closed.")
                            continue
                        open_n += 1
                        idx = None if pr is None else (0 if pr["pick"] == a else 1)
                        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
                        pick = c1.radio("Winner", [a, h], index=idx, horizontal=True, key=f"pk_{gid}",
                                        format_func=lambda t: f"{t} – {NAMES.get(t, t).split()[-1]}")
                        sa = c2.number_input(f"{a} goals", 0, 15, value=int(pr["score_away"]) if pr is not None and pd.notna(pr["score_away"]) else None,
                                             key=f"pk_sa_{gid}", placeholder="–")
                        sh = c3.number_input(f"{h} goals", 0, 15, value=int(pr["score_home"]) if pr is not None and pd.notna(pr["score_home"]) else None,
                                             key=f"pk_sh_{gid}", placeholder="–")
                        if pr is not None and bt is not None:
                            st.caption(f"🤖 The Bot picked **{bt['pick']}** ({pct1(bt['bot_p'])})"
                                       + (f", {int(bt['score_away'])}–{int(bt['score_home'])}" if pd.notna(bt["score_away"]) else ""))
                        if pick:
                            p_bot = None
                            if gid in preds:
                                ph = preds[gid]["p_home"]
                                p_bot = ph if pick == h else 1 - ph
                            if sa is not None and sh is not None and (sa == sh or (sh > sa) != (pick == h)):
                                if sa != sh:
                                    dropped.append(f"{a} @ {h}")
                                sa = sh = None          # no ties in hockey, and the score has to match the pick
                            rows.append(pick_row(name, dict(id=gid, date=day_str, start=g.get("startTimeUTC", ""),
                                                            away=a, home=h), pick, sa, sh, p_bot))
                go_ = st.form_submit_button("🔒 Lock in my picks", type="primary", width="stretch",
                                            disabled=not (name and locked_ok and open_n))
            if go_:
                if not rows:
                    st.warning("Pick at least one winner first.")
                else:
                    try:
                        n = save_picks(rows, f"{name}: {len(rows)} pick(s) for {day_str}")
                        st.toast(f"🔒 {n} pick(s) locked in. Good luck!" if n else "No changes to save.")
                        if dropped:
                            ss["pk_dropped"] = dropped
                        if n:
                            ss["pk_party"] = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Couldn't save your picks ({e}).")
            if ss.pop("pk_party", False):
                st.balloons()
            if ss.get("pk_dropped"):
                st.warning("Score guess skipped for " + ", ".join(ss.pop("pk_dropped")) +
                           ": the score has to have your pick winning (and no ties).")
            if open_n:
                st.caption("You can change picks until each game starts. The Bot's pick for a game is revealed once "
                           "you've locked in yours.")

    lb = pickem_table(df)
    with t_lb:
        if lb.empty:
            st.info("No graded picks yet. The leaderboard fills in after the first games finish.")
        else:
            if name and name in set(lb["Player"]) and BOT in set(lb["Player"]):
                me, b = lb[lb["Player"] == name].iloc[0], lb[lb["Player"] == BOT].iloc[0]
                diff = int(me["Points"] - b["Points"])
                kpis([("Your rank", f"#{int(me['Rank'])}", f"of {len(lb)}"),
                      ("You vs the Bot", f"{int(me['Points'])} – {int(b['Points'])}",
                       "you're ahead!" if diff > 0 else ("tied" if diff == 0 else f"{-diff} pts behind"),
                       "up" if diff > 0 else ("down" if diff < 0 else "")),
                      ("Your win %", f"{me['Win %']:.0f}%", f"Bot {b['Win %']:.0f}%"),
                      ("Streak", f"{int(me['Streak'])}", f"best {int(me['Best streak'])}"),
                      ("Nights you beat the Bot", f"{int(me['Nights beat the Bot'] or 0)}", "")])
                if diff > 0:
                    st.markdown('<div class="trophy">🏆 You\'re out-picking the Bot this season!</div>', unsafe_allow_html=True)
            st.dataframe(lb, hide_index=True, width="stretch", column_config={
                "Win %": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                "Nights beat the Bot": st.column_config.NumberColumn(format="%d")})
            g = df[df["status"].isin(["Won", "Lost"])].copy()
            g["points"] = pd.to_numeric(g["points"], errors="coerce").fillna(0)
            leaders = list(lb["Player"].head(6)) + ([BOT] if BOT not in set(lb["Player"].head(6)) else [])
            if name and name not in leaders and name in set(lb["Player"]):
                leaders.append(name)
            cum = g[g["player"].isin(leaders)].groupby(["date", "player"])["points"].sum().unstack(fill_value=0).cumsum()
            if len(cum) >= 2:
                fig = go.Figure()
                for p in cum.columns:
                    fig.add_trace(go.Scatter(x=cum.index, y=cum[p], name=p, mode="lines+markers",
                                             line=dict(width=4 if p in (name, BOT) else 2, dash="dot" if p == BOT else None)))
                fig.update_layout(title="Season race", yaxis_title="Points", margin=dict(t=40),
                                  xaxis=dict(tickformat="%b %d", dtick=86400000 * max(1, len(cum) // 8)),
                                  legend=dict(orientation="h", y=-0.2))
                chart(fig, "pk_race", 340)

    with t_mine:
        if not name:
            st.info("Type your name at the top to see your picks.")
        else:
            mine = df[df["player"] == name].sort_values(["date", "start_utc"], ascending=False)
            if mine.empty:
                st.info("No picks yet. Make some on the 🗳️ tab!")
            else:
                recent_wins = mine[(mine["status"] == "Won")]
                seen = ss.setdefault("pk_celebrated", set())
                cut_day = (local_today() - dt.timedelta(days=1)).isoformat()
                fresh = [pid for pid, d in zip(recent_wins["pick_id"], recent_wins["date"])
                         if pid not in seen and (ss.get("pk_seen_once") or str(d) >= cut_day)]
                if fresh:
                    confetti()
                    st.success(f"🎉 {len(fresh)} new winning pick(s)!")
                seen.update(recent_wins["pick_id"])
                ss["pk_seen_once"] = True
                view = pd.DataFrame({"Date": mine["date"], "Game": mine["away"] + " @ " + mine["home"],
                                     "Your pick": mine["pick"],
                                     "Score guess": [f"{int(a)}–{int(b)}" if pd.notna(a) else "" for a, b in
                                                     zip(mine["score_away"], mine["score_home"])],
                                     "Final": [f"{int(a)}–{int(b)}" if pd.notna(a) else "" for a, b in
                                               zip(mine["final_away"], mine["final_home"])],
                                     "Result": mine["status"].replace({"Open": "⏳"}), "Points": mine["points"],
                                     "How": mine["note"]})
                st.dataframe(view, hide_index=True, width="stretch",
                             column_config={"Points": st.column_config.NumberColumn(format="%d")})

    with t_how:
        st.markdown("**Badges you can earn**")
        mine_b = lb[lb["Player"] == name]["Badges"].iloc[0] if name and not lb.empty and name in set(lb["Player"]) else ""
        for icon, title, how in BADGES:
            got = icon in mine_b
            st.markdown(f"{'✅' if got else '⬜'} {icon} **{title}** — {how}")
        st.caption("Playing with friends: share the app link with `?player=TheirName` on the end, or they just type "
                   "their name. To keep strangers out, add `PICKEM_CODE = \"your-code\"` to your Streamlit secrets.")


# ---------------- races ----------------
def page_races():
    top = st.columns([3, 2], vertical_alignment="bottom")
    top[0].markdown("## 🏁 Races")
    with top[1]:
        settings_popover()
    t_sc, t_cup = st.tabs(["🎯 Scoring race", "🏆 Championship race"])
    with t_sc:
        stat = st.segmented_control("Race", ["Points", "Goals"], default="Points", key="race_stat") or "Points"
        with st.spinner("Simulating the rest of the season 20,000 times…"):
            try:
                sr = scoring_race(stat)
            except Exception as e:
                sr = pd.DataFrame()
                st.error(f"Couldn't load player stats ({e}).")
        if not sr.empty:
            lead = sr.iloc[0]
            kpis([(f"Favourite for the {stat.lower()} title", lead["Player"], f"{lead['Team']} · {pct1(lead['Title %'] / 100)}"),
                  (f"{stat} leader now", sr.sort_values(stat, ascending=False).iloc[0]["Player"],
                   f"{int(sr[stat].max())} {stat.lower()}"),
                  ("Games left (most)", f"{int(sr['Games left'].max())}", "for a contender")])
            shots = {str(p["id"]): p.get("headshot", "") for lst in league_players()[0].values() for p in lst}
            view = sr.drop(columns=["id"]).copy()
            view.insert(0, "Photo", [shots.get(str(i), "") for i in sr["id"]])
            st.dataframe(view, hide_index=True, width="stretch", height=560, column_config={
                "Photo": st.column_config.ImageColumn(width="small"),
                "Per game": st.column_config.NumberColumn(format="%.2f"),
                "Projected": st.column_config.NumberColumn(format="%.1f"),
                "Title %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                "Top 3 %": st.column_config.NumberColumn(format="%.0f%%")})
            st.caption("Each player's pace is blended with the group's pace (hot starts cool off a bit), and every "
                       "player misses about 1 game in 12 on average.")
            st.markdown(f"**🏎️ The {stat.lower()} race so far**")
            ids = tuple(str(i) for i in sr.sort_values(stat, ascending=False)["id"].head(8))
            with st.spinner("Loading game logs…"):
                seqs = points_by_date(ids, stat)
            names = dict(zip(sr["id"].astype(str), sr["Player"]))
            teams = dict(zip(sr["id"].astype(str), sr["Team"]))
            dates = sorted({d for s in seqs.values() for d, _ in s if d})
            if len(dates) >= 2:
                weeks = sorted({(pd.Timestamp(d) - pd.Timedelta(days=pd.Timestamp(d).weekday())).date().isoformat()
                                for d in dates}) + [dates[-1]]

                def at(pid, day):
                    v = 0
                    for d, t in seqs.get(pid, []):
                        if d <= day:
                            v = t
                    return v

                def bar(day):
                    vals = sorted(((at(p, day), p) for p in ids), reverse=True)
                    return go.Bar(x=[v for v, _ in vals], y=[names.get(p, p) for _, p in vals], orientation="h",
                                  marker_color=[TEAM_COLORS.get(teams.get(p), "#888") for _, p in vals],
                                  text=[f"{v}" for v, _ in vals], textposition="outside", cliponaxis=False)
                xmax = max(at(p, dates[-1]) for p in ids) * 1.15 + 1
                fig = go.Figure(data=[bar(dates[-1])],
                                frames=[go.Frame(data=[bar(w)], name=w) for w in weeks])
                fig.update_layout(
                    xaxis=dict(range=[0, xmax], title=stat), yaxis=dict(autorange="reversed"), margin=dict(t=56),
                    updatemenus=[dict(type="buttons", direction="left", showactive=False, x=0, y=1.02, xanchor="left",
                                      yanchor="bottom", pad=dict(b=6), buttons=[
                        dict(label="▶ Play", method="animate",
                             args=[None, dict(frame=dict(duration=650, redraw=True), fromcurrent=False,
                                              transition=dict(duration=400))]),
                        dict(label="⏸", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
                    sliders=[dict(active=len(weeks) - 1, y=-0.05, currentvalue=dict(prefix="Week of "),
                                  steps=[dict(label=w[5:], method="animate",
                                              args=[[w], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                                         for w in weeks])])
                chart(fig, f"race_anim_{stat}", 470)
                st.caption("Tap ▶ Play to watch the race week by week, or drag the slider.")
            else:
                st.caption("The race chart appears once players have a few games logged.")
    with t_cup:
        st.caption("Every team's chance of making the playoffs and winning the Ed Chynoweth Cup, from thousands of "
                   "simulated seasons. The app keeps one snapshot a day so you can watch the race change.")
        if league_ready("league_btn_races"):
            with st.spinner("Simulating 2,000 seasons…"):
                sres = season_sim(ss["n_recent"], 2000)
            tbl = sres["table"]
            day_key = f"_odds_saved_{local_today().isoformat()}"
            if not ss.get(day_key):
                ss[day_key] = True
                try:
                    save_title_odds(tbl)
                except Exception:
                    pass
            topc = tbl.sort_values("Cup %", ascending=False).head(12)
            fig = go.Figure(go.Bar(x=topc["Cup %"], y=topc["Team"], orientation="h",
                                   marker_color=[TEAM_COLORS.get(t, "#888") for t in topc["Team"]],
                                   text=[f"{v:.1f}%" for v in topc["Cup %"]], textposition="outside", cliponaxis=False))
            fig.update_layout(title="Ed Chynoweth Cup odds", yaxis=dict(autorange="reversed"), xaxis_title="%",
                              margin=dict(t=40))
            chart(fig, "cup_race_now", 420)
            hist = load_store("odds_history")
            if hist["date"].nunique() >= 2:
                metric = st.segmented_control("Show", ["Champion %", "Playoffs %", "Projected points"],
                                              default="Champion %", key="cup_hist_metric") or "Champion %"
                col = {"Champion %": "champ", "Playoffs %": "playoffs", "Projected points": "proj_pts"}[metric]
                last = hist[hist["date"] == hist["date"].max()]
                keep = last.sort_values(col, ascending=False)["team"].head(8).tolist()
                fig = go.Figure()
                for t in keep:
                    h_ = hist[hist["team"] == t].sort_values("date")
                    fig.add_trace(go.Scatter(x=h_["date"], y=h_[col], name=t, mode="lines+markers",
                                             line=dict(color=TEAM_COLORS.get(t, "#888"), width=3)))
                fig.update_layout(title=f"{metric} over time", margin=dict(t=40), legend=dict(orientation="h", y=-0.2))
                chart(fig, "cup_race_hist", 380)
                first_day = sorted(hist["date"].unique())[max(0, hist["date"].nunique() - 8)]
                a_ = hist[hist["date"] == first_day].set_index("team")[col]
                b_ = last.set_index("team")[col]
                mv = (b_ - a_).dropna().sort_values()
                if len(mv):
                    st.markdown(f"**Biggest movers since {first_day}:** 📈 {mv.index[-1]} {mv.iloc[-1]:+.1f} · "
                                f"📉 {mv.index[0]} {mv.iloc[0]:+.1f}")
            else:
                st.caption("📅 Come back tomorrow: once there are two daily snapshots, you'll see how every team's "
                           "odds move over time (the daily robot saves one every morning).")


# ---------------- rivalries ----------------
def page_rivalry():
    c = st.columns([3, 3, 2], vertical_alignment="bottom")
    a = c[0].selectbox("Team", TEAMS, index=TEAMS.index(ss["away"]), format_func=_label, key="riv_a")
    b = c[1].selectbox("Rival", TEAMS, index=TEAMS.index(ss["home"]), format_func=_label, key="riv_b")
    with c[2]:
        settings_popover()
    if a == b:
        st.warning("Pick two different teams.")
        return
    ch, ca = team_colors(b, a)
    games = meetings(a, b)
    done = [g for g in games if g["state"] == "FINAL"]
    upcoming = [g for g in games if g["state"] != "FINAL" and g["season"] == "This season"
                and g["date"] >= local_today().isoformat()]
    team_hero(a, b, f"⚔️ {len(done)} meetings over the last two seasons · {len(upcoming)} still to come this season")
    if not done and not upcoming:
        st.info("These two haven't met in the last two seasons and aren't scheduled to.")
        return

    def rec(lst, team):
        w = sum(1 for g in lst if (g["hs"] > g["as_"]) == (g["home"] == team))
        return w, len(lst) - w

    this = [g for g in done if g["season"] == "This season"]
    wa, la = rec(done, a)
    wt, lt = rec(this, a)
    gf = sum(g["hs"] if g["home"] == a else g["as_"] for g in done)
    ga_ = sum(g["as_"] if g["home"] == a else g["hs"] for g in done)
    streak, who = 0, None
    for g in reversed(done):
        w = g["home"] if g["hs"] > g["as_"] else g["away"]
        if who in (None, w):
            who, streak = w, streak + 1
        else:
            break
    kpis([(f"{a} this season", f"{wt}-{lt}", f"vs {b}"), (f"{a} last two seasons", f"{wa}-{la}", ""),
          ("Goals", f"{gf} – {ga_}", f"{a} – {b}"),
          ("Went to OT/SO", f"{sum(1 for g in done if g['end'] != 'REG')}", f"of {len(done)}"),
          ("Streak", f"{who} {streak}" if who else "–", "in a row")])
    if done:
        x = [f"{g['date'][5:]}{'*' if g['season'] == 'Last season' else ''}" for g in done]
        m = [(g["hs"] - g["as_"]) * (1 if g["home"] == a else -1) for g in done]
        fig = go.Figure(go.Bar(x=x, y=m, marker_color=[ca if v > 0 else ch for v in m],
                               text=[f"{(g['as_'] if g['home'] == b else g['hs'])}–{(g['hs'] if g['home'] == b else g['as_'])}"
                                     + ("" if g["end"] == "REG" else f" {g['end']}") for g in done],
                               textposition="outside", cliponaxis=False))
        fig.update_layout(title=f"Every meeting (above zero = {a} won, * = last season)",
                          yaxis_title=f"{a} goal margin", margin=dict(t=40))
        chart(fig, "riv_margin", 320)
    left, right = st.columns(2)
    with left:
        st.markdown("**🗓️ Results**")
        if not done:
            st.caption("They haven't played yet this season or last.")
        else:
            st.dataframe(pd.DataFrame([{"Date": g["date"], "Season": g["season"], "Game": f"{g['away']} @ {g['home']}",
                                        "Score": f"{g['as_']}–{g['hs']}" + ("" if g["end"] == "REG" else f" ({g['end']})"),
                                        "Winner": g["home"] if g["hs"] > g["as_"] else g["away"]} for g in reversed(done)]),
                         hide_index=True, width="stretch", height=300)
    with right:
        st.markdown("**🔥 Top scorers in this rivalry**")
        try:
            pb = fetch_pbps([g["id"] for g in done])
        except Exception:
            pb = []
        tally, info = Counter(), {}
        for g in pb:
            for pid, v in (g.get("pstats") or {}).items():
                tally[pid] += v[1] + v[2]
                tally[(pid, "g")] += v[1]
            info.update(g.get("rinfo") or {})
            for pid, t_ in g["roster"]:
                info.setdefault(pid, ("F", str(pid)))
                info[(pid, "team")] = g["home"] if t_ == g["home_id"] else g["away"]
        top_ = [(pid, n) for pid, n in tally.most_common(40) if not isinstance(pid, tuple) and n > 0][:8]
        if top_:
            st.dataframe(pd.DataFrame([{"Player": info.get(pid, ("", pid))[1], "Team": info.get((pid, "team"), ""),
                                        "G": tally[(pid, "g")], "PTS": n} for pid, n in top_]),
                         hide_index=True, width="stretch", height=300)
        else:
            st.caption("No scoring details yet.")
    if upcoming:
        g = upcoming[0]
        st.markdown(f"**🔮 Next meeting: {g['away']} @ {g['home']} · {pd.Timestamp(g['date']):%a %b %d}**")
        try:
            hp, ap = build_profile(g["home"], ss["n_recent"]), build_profile(g["away"], ss["n_recent"])
            gs = starting_goalies(g["home"], g["away"], g["date"], hp, ap)
            r = run_matchup(g["home"], g["away"], gs["home"][0], gs["away"][0], default_lineup(hp, False)[1],
                            default_lineup(ap, False)[1], 3000, False, ss["n_recent"], None, g["date"], False)
            s0 = r["top_scores"][0][0]
            kpis([(f"{g['away']} win", pct1(r["away_win"]), ""), (f"{g['home']} win", pct1(r["home_win"]), ""),
                  ("Most likely", f"{s0[1]}–{s0[0]}", f"{g['away']} – {g['home']}"),
                  ("Goes to OT/SO", pct1(r["ot"] + r["so"]), "")])
            if st.button("Open it in Game Center ›", key="riv_open"):
                open_game(g["home"], g["away"], dict(id=g["id"], date=g["date"], start=g["start"], playoff=False))
                st.switch_page(PAGE["game"])
        except Exception as e:
            st.caption(f"Couldn't simulate the next meeting ({e}).")
    st.markdown("**📏 Tale of the tape**")
    try:
        ra, rb = STN.loc[a], STN.loc[b]
        pa, pb_ = build_profile(a, ss["n_recent"]), build_profile(b, ss["n_recent"])
        rows = [("Record", f"{ra.w}-{ra.l}-{ra.otl}", f"{rb.w}-{rb.l}-{rb.otl}"), ("Points", ra.pts, rb.pts),
                ("Goals for / against per game", f"{ra.gf / max(ra.gp, 1):.2f} / {ra.ga / max(ra.gp, 1):.2f}",
                 f"{rb.gf / max(rb.gp, 1):.2f} / {rb.ga / max(rb.gp, 1):.2f}"),
                ("Last 10 · streak", f"{ra.l10} · {ra.streak}", f"{rb.l10} · {rb.streak}"),
                ("Recent shots for / against", f"{pa['sog_pg']:.1f} / {pa['soga_pg']:.1f}", f"{pb_['sog_pg']:.1f} / {pb_['soga_pg']:.1f}"),
                ("Recent PP% / PK%", f"{pct(pa['pp_pct'])} / {pct(pa['pk_pct'])}", f"{pct(pb_['pp_pct'])} / {pct(pb_['pk_pct'])}")]
        km = arena_km(a, b)
        if km:
            rows.append(("Bus trip between the rinks", f"{km:,.0f} km", ""))
        st.dataframe(pd.DataFrame(rows, columns=["", a, b]).astype(str), hide_index=True, width="stretch")
    except Exception:
        pass


# ==========================================================================
# Daily auto-save (run by the GitHub robot: WHL_AUTOSAVE=1 python whl_app.py)
# ==========================================================================
def autosave():
    print(f"WHL auto-save · {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC · storage: "
          f"{'GitHub ' + _gh_conf()['repo'] if _gh_conf() else 'local files'}")
    for name, fn in (("games", grade_tracker), ("players", grade_players), ("bets", grade_bets),
                     ("pick'em picks", grade_pickem)):
        try:
            print(f"  graded {fn()[1]} {name}")
        except Exception as e:
            print(f"  couldn't grade {name}: {e}")
    day = local_today().isoformat()
    if os.environ.get("WHL_TITLE_ODDS") == "1":
        try:
            sres = season_sim(15, 2000)
            print(f"  saved title odds for {save_title_odds(sres['table'])} teams")
        except Exception as e:
            print(f"  couldn't save title odds: {e}")
    games = [g for g in day_games_safe(day) if not _started(g.get("startTimeUTC", ""))]
    if not games:
        print(f"  no games left to predict on {day}")
        return
    try:
        saved, saved_p = simulate_slate(day, games, int(os.environ.get("WHL_AUTOSAVE_SIMS", "3000")), True)
        print(f"  {day}: saved {saved} game and {saved_p} player prediction set(s) for {len(games)} upcoming game(s)")
        for gid, r in ss.get("slate", {}).get(day, {}).items():
            if "error" in r:
                print(f"    {r['away']} @ {r['home']}: {r['error']}")
            else:
                print(f"    {r['away']} @ {r['home']}: {r['home']} {100 * r['p_home']:.0f}% · {r['ag']} vs {r['hg']}"
                      + (f" · {r['fat']}" if r.get("fat") else ""))
        try:
            print(f"  bot made {save_picks(bot_rows(day, day_predictions(day)), f'Bot picks for {day}')} pick'em pick(s)")
        except Exception as e:
            print(f"  couldn't save the bot's picks: {e}")
    except Exception as e:
        print(f"  auto-save failed: {e}")
        raise


if os.environ.get("WHL_AUTOSAVE"):
    autosave()
    raise SystemExit(0)


# ==========================================================================
# Navigation (top bar on computers, menu on phones)
# ==========================================================================
_pages = [st.Page(page_today, title="Games", icon="📅", url_path="games", default=True),
          st.Page(page_game, title="Game Center", icon="🏒", url_path="game"),
          st.Page(page_pickem, title="Beat the Bot", icon="🎮", url_path="pickem"),
          st.Page(page_players, title="Players", icon="⛸️", url_path="players"),
          st.Page(page_bets, title="My Bets", icon="💰", url_path="bets"),
          st.Page(page_tracker, title="Tracker", icon="📈", url_path="tracker"),
          st.Page(page_races, title="Races", icon="🏁", url_path="races"),
          st.Page(page_season, title="Season", icon="🏆", url_path="season"),
          st.Page(page_rivalry, title="Rivalries", icon="⚔️", url_path="rivalry"),
          st.Page(page_teams, title="Teams", icon="📊", url_path="teams")]
PAGE = {p.url_path: p for p in _pages}
_nav = st.navigation(_pages, position="top")
_nav.run()
st.caption("Official statistics provided by the Western Hockey League (HockeyTech) · "
           "simulations are estimates, not guarantees.")