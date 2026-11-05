#!/usr/bin/env python3
"""Write one match of a built dataset in the challenge's file format, as if it were the real bundle.

    python tools/export_challenge_format.py --datasets external/datasets_v2_1 --match METRICA1 --out fake_data/export

Writes <out>/tactical/BAR-ATH_tactical_atd.xml, BAR-ATH_tactical_atd.txt and
BAR-ATH_tactical_smart_tagging.xml -- the names games.asset_paths() expects -- so every reader of
the real bundle (kloppy through starter/load_atd.py, starter/load_smart_tagging.py,
tools/heuristic_pipeline.py) reads a bank match unchanged. tests/test_challenge_roundtrip.py
checks that what they read is what the dataset holds.

The ATD follows the real export: one line per video frame, "frame_id:x,y,s;...;x,y,s:x,y,s" with
frame_id = video_frame + 1, coordinates normalised to 0-1 by the match's own pitch, NaN where an
object is missing, and speeds in m/s from consecutive positions. Slot k is "Track k+1" (shirt
number k+1); the first 23 belong to the home team, which is listed first. The Smart Tagging file
is the dataset's own, already in the SportsCode format of the real one.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "tools"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import games  # noqa: E402
from build_datasets import FPS, load_match  # noqa: E402

TEAM_IDS = ("home", "away")


def speeds(xy):
    """m/s from consecutive positions [n, ..., 2]; NaN where either frame is missing."""
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        out[1:] = np.linalg.norm(np.diff(xy.astype(np.float64), axis=0), axis=-1) * FPS
    return out


def write_atd_meta(path, meta, slot_team, home_left_first):
    n = meta["n_frames"]
    halves = [(a + 1, b + 1) for _, a, b in meta["halves"]]                # frame_id = video_frame + 1
    names = (meta["home"], meta["away"])
    directions = ("left_to_right", "right_to_left") if home_left_first else ("right_to_left", "left_to_right")

    def param(name, value):
        return "<ProviderParameter><Name>%s</Name><Value>%s</Value></ProviderParameter>" % (name, value)

    globals_ = "".join([param("first_half_start", halves[0][0]), param("first_half_end", halves[0][1])] +
                       ([param("second_half_start", halves[1][0]), param("second_half_end", halves[1][1])]
                        if len(halves) > 1 else []) +
                       [param("data_type", "atd"), param("tracking_source", "exported bank match")])
    teams = "".join('<Team id="%s"><Name>%s</Name><ProviderTeamsParameters>%s</ProviderTeamsParameters></Team>'
                    % (TEAM_IDS[t], names[t], param("attack_direction_first_half", directions[t])) for t in (0, 1))
    players, channels, refs = [], [], []
    for k, t in enumerate(slot_team.tolist()):
        pid = str(k + 1)
        players.append('<Player id="%s" teamId="%s"><Name>Track %s</Name><ShirtNumber>%s</ShirtNumber>'
                       '<ProviderPlayerParameters>%s%s</ProviderPlayerParameters></Player>'
                       % (pid, TEAM_IDS[t], pid, pid, param("position_type", "Unknown"), param("position_index", -1)))
        for ch, name in (("x", "x"), ("y", "y"), ("s", "speed")):
            channels.append('<PlayerChannel channelId="%s" id="player%s_%s" playerId="%s"/>' % (ch, pid, name, pid))
        refs.append('<SplitRegister separator=",">%s</SplitRegister>' % "".join(
            '<PlayerChannelRef playerChannelId="player%s_%s"/>' % (pid, name) for name in ("x", "y", "speed")))
    L, W = meta["pitch_m"]
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n<main><Metadata>'
        '<GlobalConfig><FileName>%s</FileName><TrackingType>Optical</TrackingType>'
        '<ProviderName>Metrica Sports</ProviderName><FrameRate>%d</FrameRate>'
        '<ProviderGlobalParameters>%s</ProviderGlobalParameters></GlobalConfig>'
        '<Sessions><Session id="session1"><SessionType>Match</SessionType><MatchParameters>'
        '<Score idLocalTeam="home" idVisitingTeam="away"><LocalTeamScore/><VisitingTeamScore/></Score>'
        '</MatchParameters><FieldSize><Width>%d</Width><Height>%d</Height></FieldSize></Session></Sessions>'
        '<Teams>%s</Teams><Players>%s</Players>'
        '<Devices><Device id="device1"><Name>Device 1</Name><Sensors>'
        '<Sensor id="position"><Name>Position</Name><Channels>'
        '<Channel id="x"><Name>position_x</Name><Unit>normalized</Unit></Channel>'
        '<Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel></Channels></Sensor>'
        '<Sensor id="speed"><Name>Speed in metres per second</Name><Channels>'
        '<Channel id="s"><Name>speed</Name><Unit>meters per second</Unit></Channel></Channels></Sensor>'
        '</Sensors></Device></Devices><PlayerChannels>%s</PlayerChannels></Metadata>'
        '<DataFormatSpecifications><DataFormatSpecification endFrame="%d" separator=":" startFrame="1">'
        '<StringRegister name="frameCount"/><SplitRegister separator=";">%s</SplitRegister>'
        '<SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/>'
        '<BallChannelRef channelId="s"/></SplitRegister></DataFormatSpecification>'
        '</DataFormatSpecifications></main>\n'
        % (meta["match"], FPS, globals_, round(L), round(W), teams, "".join(players), "".join(channels), n,
           "".join(refs)))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(xml)


def write_atd_raw(path, players, ball, pitch):
    """players [n, slots, 2] and ball [n, 2] in metres -> the raw ATD text, normalised 0-1."""
    L, W = pitch
    scale = np.array([L, W])
    norm_p, norm_b = players / scale, ball / scale
    sp_p, sp_b = speeds(players), speeds(ball)
    empty = "NaN,NaN,NaN"

    def group(xy, s):
        if xy[0] != xy[0]:
            return empty
        return "%.6f,%.6f,%s" % (xy[0], xy[1], "NaN" if s != s else "%.3f" % s)

    buffer = []
    with open(path, "w", newline="\n") as handle:
        for f in range(len(ball)):
            row = norm_p[f].tolist()
            spd = sp_p[f].tolist()
            buffer.append("%d:%s:%s\n" % (f + 1, ";".join(group(xy, s) for xy, s in zip(row, spd)),
                                          group(norm_b[f].tolist(), float(sp_b[f]))))
            if len(buffer) >= 5000:
                handle.writelines(buffer)
                buffer = []
        handle.writelines(buffer)


def export(datasets, match, seed, out):
    """Write the match in challenge format under `out`; returns games.asset_paths for it."""
    m = load_match(datasets, match, seed)
    meta = m["meta"]
    paths = games.asset_paths(games.GAME, out)
    os.makedirs(os.path.dirname(paths["atd_raw"]), exist_ok=True)
    _, a, b = meta["halves"][0]
    with np.errstate(invalid="ignore"):
        home_x = np.nanmean(m["players"][a:b + 1][:, m["slot_team"] == 0, 0])
    write_atd_meta(paths["atd_meta"], meta, m["slot_team"], home_x < meta["pitch_m"][0] / 2)
    write_atd_raw(paths["atd_raw"], m["players"], m["ball"], meta["pitch_m"])
    shutil.copyfile(m["smart_tagging"], paths["smart_tagging"])
    with open(os.path.join(out, "exported_from.json"), "w") as handle:
        json.dump({"datasets": os.path.abspath(datasets), "match": match, "seed": seed}, handle, indent=2)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", default=os.path.join(ROOT, "external", "datasets_v2_1"))
    parser.add_argument("--match", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="folder that plays the role of the data bundle")
    args = parser.parse_args()
    paths = export(args.datasets, args.match, args.seed, args.out)
    print("\n".join("%-14s %s" % kv for kv in paths.items() if os.path.isfile(kv[1])))


if __name__ == "__main__":
    main()
