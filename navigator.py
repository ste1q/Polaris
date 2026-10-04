#!/usr/bin/env python3
"""
Voice-guided indoor navigation for the wristband (v2).

Install:
  python3 -m pip install pyserial python-dotenv SpeechRecognition pyaudio elevenlabs

Loads .env from WRISTBAND_ENV, /Users/s/Downloads/wristband/.env, next to this file, or the current folder (first found):
  ARDUINO_PORT=/dev/cu.usbmodemXXXX
  ARDUINO_BAUD=115200
  ELEVENLABS_API_KEY=...
  ELEVENLABS_VOICE_ID=...
  CROWD_FILE=crowd.json          # written by crowd_monitor.py
  RESCAN_SECS=15                 # holding the sensor on one marker this long counts as a new scan

Run:
  python3 navigator.py                              # real wristband + microphone
  python3 navigator.py --simulate                   # no hardware: type events (see below)
  python3 navigator.py --test-route room_195 room_197 [--avoid-stairs] [--crowd bathroom=0.9] [--speak]
  python3 navigator.py --test-voice                 # checks .env + ElevenLabs by speaking a sentence

Hardware protocol (matches the supplied Arduino sketch exactly, which is NOT modified):
  Arduino -> PC   WRISTBAND_READY
                  DATA,r,g,b,c            raw TCS34725 values every 2.5 s
                  BUZZER_ON_OK / BUZZER_OFF_OK
  PC -> Arduino   BUZZER_ON | BUZZER_OFF | BEEP
Because the sketch sends raw values, color classification happens here, using the same
detectColor() logic as the sketch.

Buzzer cues (the only tactile/audio output the sketch supports):
  2 short beeps = turn left     3 short beeps = turn right     1 long beep = straight
  4 rapid beeps = turn around   long + 2 short = arrived

Simulate-mode commands (type and press Enter):
  marker bathroom      pretend the wristband touched the bathroom's marker
  anything else        treated as something you "said", e.g. "take me to room 197"
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional
    def load_dotenv(*_a, **_k):
        return False

HERE = Path(__file__).resolve().parent
ENV_CANDIDATES = (
    os.getenv("WRISTBAND_ENV"),
    "/Users/s/Downloads/wristband/.env",
    HERE / ".env",
    Path.cwd() / ".env",
)
ENV_LOADED = None
for _env in ENV_CANDIDATES:
    if _env and Path(_env).exists():
        load_dotenv(_env)
        ENV_LOADED = str(_env)
        break

BUILDING_FILE = HERE / "building.json"
ARDUINO_PORT = os.getenv("ARDUINO_PORT", "").strip()
ARDUINO_BAUD = int(os.getenv("ARDUINO_BAUD", "115200"))
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "").strip()
CROWD_FILE = Path(os.getenv("CROWD_FILE", str(HERE / "crowd.json")))
RESCAN_SECS = float(os.getenv("RESCAN_SECS", "15"))  # same-color scan accepted again after this long

# Routing / behaviour tuning
CROWD_WEIGHT = 2.0          # a fully crowded node makes the edge into it 3x as costly
STAIR_PENALTY_M = 50.0      # extra "meters" when --avoid-stairs is on
CROWD_WARN_LEVEL = 0.6      # say "crowded" at/above this level
CROWD_STALE_SECS = 30.0     # ignore camera data older than this
CROWD_CHECK_SECS = 5.0
REMIND_SECS = 25.0          # repeat guidance if no marker seen for this long


# --------------------------------------------------------------------------- #
# Building model
# --------------------------------------------------------------------------- #
class Building:
    def __init__(self, path: Path):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.name: str = data["building"]
        self.calibrated: bool = data.get("calibrated", True)
        self.nodes: dict = data["nodes"]
        # graph[a][b] = {"distance": meters, "heading": degrees or None}
        self.graph: dict[str, dict[str, dict]] = {n: {} for n in self.nodes}
        for e in data["edges"]:
            a, b = e["from"], e["to"]
            dist = float(e.get("distance_m", 10))
            h = e.get("heading_deg")
            self.graph[a][b] = {"distance": dist, "heading": h}
            self.graph[b][a] = {"distance": dist,
                                "heading": None if h is None else (h + 180) % 360}

        self.marker_to_nodes: dict[str, list[str]] = {}
        for nid, info in self.nodes.items():
            color = info.get("marker_color") or info.get("color")
            self.marker_to_nodes.setdefault(color, []).append(nid)
        self.marker_colors = set(self.marker_to_nodes)
        self.start_node: str = next(iter(self.nodes))  # first node listed = starting position

        self.aliases: dict[str, str] = {}
        for nid, info in self.nodes.items():
            names = {info["label"].lower(), nid.replace("_", " ")}
            names.update(a.lower() for a in info.get("aliases", []))
            for n in names:
                self.aliases[n] = nid

    def label(self, node: str) -> str:
        return self.nodes[node]["label"]

    def shared_markers(self) -> dict[str, list[str]]:
        return {c: n for c, n in self.marker_to_nodes.items() if len(n) > 1}

    def color_of(self, node: str) -> str:
        return self.nodes[node].get("marker_color") or self.nodes[node].get("color")

    def node_for_marker(self, color: str, current: Optional[str],
                        expected: Optional[str]) -> Optional[str]:
        """Which node did the user just scan?

        Unique colors identify the node directly. For a shared color (e.g. three blue
        rooms) we rely on the guided order: the user was told to walk to `expected`
        and scan, so a matching color means they arrived there. Otherwise it is a
        re-scan of `current`, or else the single neighbor of `current` with that color.
        """
        cands = self.marker_to_nodes.get(color, [])
        if len(cands) == 1:
            return cands[0]
        if expected in cands:
            return expected
        if current in cands:
            return current
        adj = [n for n in cands if current in self.graph and n in self.graph[current]]
        return adj[0] if len(adj) == 1 else None

    @staticmethod
    def normalize_numbers(t: str) -> str:
        """'one nine seven' / '1 9 7' / 'one ninety seven' -> '197'."""
        words = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                 "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
        t = re.sub(r"\bone (?:hundred (?:and )?)?ninety[- ]?(\w+)\b",
                   lambda m: "19" + words[m.group(1)] if m.group(1) in words else m.group(0), t)
        t = re.sub(r"\b(" + "|".join(words) + r")\b", lambda m: words[m.group(1)], t)
        prev = None
        while prev != t:  # join single digits separated by spaces: "1 9 7" -> "197"
            prev = t
            t = re.sub(r"\b(\d) (?=\d\b)", r"\1", t)
        return t

    def find_node(self, text: str) -> Optional[str]:
        t = self.normalize_numbers(re.sub(r"\s+", " ", text.lower()))
        for phrase in sorted(self.aliases, key=len, reverse=True):  # longest first
            if re.search(r"\b" + re.escape(phrase) + r"\b", t):
                return self.aliases[phrase]
        return None

    # ---- routing ----
    def edge_cost(self, a: str, b: str, crowd: dict[str, float],
                  avoid_stairs: bool, goal: Optional[str] = None) -> float:
        cost = self.graph[a][b]["distance"] * (1 + CROWD_WEIGHT * crowd.get(b, 0.0))
        if avoid_stairs and self.nodes[b]["type"] == "staircase" and b != goal:
            cost += STAIR_PENALTY_M
        return cost

    def plan(self, start: str, goal: str, crowd: Optional[dict[str, float]] = None,
             avoid_stairs: bool = False) -> list[str]:
        """Dijkstra over distance, inflated by crowding and (optionally) staircases."""
        crowd = crowd or {}
        if start not in self.graph or goal not in self.graph:
            raise ValueError(f"Unknown start/goal: {start!r}, {goal!r}")
        best = {start: 0.0}
        prev: dict[str, str] = {}
        pq = [(0.0, start)]
        while pq:
            cost, u = heapq.heappop(pq)
            if u == goal:
                break
            if cost > best.get(u, float("inf")):
                continue
            for v in self.graph[u]:
                c = cost + self.edge_cost(u, v, crowd, avoid_stairs, goal)
                if c < best.get(v, float("inf")):
                    best[v] = c
                    prev[v] = u
                    heapq.heappush(pq, (c, v))
        if goal not in best:
            raise ValueError(f"No route from {start} to {goal}")
        route = [goal]
        while route[-1] != start:
            route.append(prev[route[-1]])
        return route[::-1]

    def route_cost(self, route: list[str], crowd, avoid_stairs) -> float:
        goal = route[-1]
        return sum(self.edge_cost(a, b, crowd, avoid_stairs, goal)
                   for a, b in zip(route, route[1:]))

    def route_meters(self, route: list[str]) -> float:
        return sum(self.graph[a][b]["distance"] for a, b in zip(route, route[1:]))


# --------------------------------------------------------------------------- #
# Instructions
# --------------------------------------------------------------------------- #
def classify_turn(heading_in: Optional[float], heading_out: Optional[float]):
    """Return (kind, side). Compass headings increase clockwise, so +diff = right."""
    if heading_in is None or heading_out is None:
        return None, None
    d = (heading_out - heading_in + 540) % 360 - 180
    a, side = abs(d), ("right" if d > 0 else "left")
    if a < 25:
        return "straight", None
    if a < 60:
        return "slight", side
    if a < 135:
        return "turn", side
    if a < 165:
        return "sharp", side
    return "uturn", None


TURN_TEXT = {
    "straight": "Continue straight",
    "slight": "Bear slightly {side}",
    "turn": "Turn {side}",
    "sharp": "Make a sharp {side}",
    "uturn": "Turn around",
}


def leg_instruction(b: Building, a: str, to: str, heading_in: Optional[float],
                    goal: str, crowd: dict[str, float]) -> tuple[str, str]:
    """Spoken text + haptic name for travelling a -> to."""
    edge = b.graph[a][to]
    kind, side = classify_turn(heading_in, edge["heading"])
    meters = max(1, round(edge["distance"]))
    unit = "meter" if meters == 1 else "meters"
    walk = f"walk about {meters} {unit} to {b.label(to)}"
    if kind:
        text = TURN_TEXT[kind].format(side=side) + f", then {walk}."
        haptic = {"straight": "STRAIGHT", "uturn": "UTURN"}.get(kind, (side or "").upper())
    else:
        text, haptic = walk[0].upper() + walk[1:] + ".", "STRAIGHT"
    if b.nodes[to]["type"] == "staircase":
        text += " Caution, staircase. Use the handrail."
    if crowd.get(to, 0.0) >= CROWD_WARN_LEVEL:
        text += f" {b.label(to)} is crowded, so go slowly."
    return text, haptic


def describe_route(b: Building, route: list[str], heading: Optional[float] = None,
                   crowd: Optional[dict[str, float]] = None) -> list[str]:
    crowd = crowd or {}
    if len(route) == 1:
        return [f"You are already at {b.label(route[0])}."]
    out, h = [], heading
    for a, to in zip(route, route[1:]):
        out.append(leg_instruction(b, a, to, h, route[-1], crowd)[0])
        h = b.graph[a][to]["heading"]
    out.append(f"You have arrived at {b.label(route[-1])}.")
    return out


# --------------------------------------------------------------------------- #
# Speech utterance parsing
# --------------------------------------------------------------------------- #
def parse_utterance(b: Building, text: str):
    t = text.lower().strip()
    node = b.find_node(t)
    if re.search(r"\b(stop|cancel|quit|never mind)\b", t):
        return "stop", None
    if re.search(r"\b(repeat|again|say that)\b", t):
        return "repeat", None
    if re.search(r"\bwhere am i\b", t):
        return "where", None
    if node and re.search(r"\b(i am|i'm|im|i m|start(ing)?)\b.*\b(at|in|from)\b", t):
        return "here", node
    if node:
        return "goto", node
    if re.search(r"\b(take me|bring me|guide me|lead me|get me|navigate|go to|directions)\b", t):
        return "need_place", None
    return None, None


# --------------------------------------------------------------------------- #
# Fallback color classifier (only used with v1 firmware that sends raw RGBC)
# --------------------------------------------------------------------------- #
def classify_color(r: int, g: int, b: int, clear: int) -> str:
    if clear < 50:
        return "black"
    red, green, blue = r / clear, g / clear, b / clear
    mx, mn = max(red, green, blue), min(red, green, blue)
    delta = mx - mn
    if delta < 0.03:
        return "white" if clear > 500 else "gray"
    if delta / mx < 0.20:
        return "gray"
    if mx == red:
        hue = (60.0 * ((green - blue) / delta)) % 360
    elif mx == green:
        hue = 60.0 * ((blue - red) / delta + 2.0)
    else:
        hue = 60.0 * ((red - green) / delta + 4.0)
    for limit, name in ((15, "red"), (45, "orange"), (70, "yellow"), (160, "green"),
                        (200, "cyan"), (260, "blue"), (290, "purple"), (345, "pink")):
        if hue < limit:
            return name
    return "red"


COLOR_WORDS = {"red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink",
               "white", "gray", "black", "unknown"}
KNOWN_ACKS = re.compile(r"^(BUZZER_(ON|OFF)_OK|PONG|OK,.*|DEBUG.*)$")


def parse_device_line(line: str):
    """Parse one serial line into an event tuple.

    Returns ("color", name) | ("error", text) | ("info", text) | ("unknown", text) | None.
    Accepts the supplied sketch's `DATA,r,g,b,c`, plus variants (a trailing color name,
    a bare `r,g,b,c`, or a bare color word) so small firmware differences don't silence it.
    """
    line = line.strip()
    if not line:
        return None
    if line == "WRISTBAND_READY":
        return ("info", "Wristband connected.")
    if line.upper().startswith("ERROR"):
        return ("error", line)
    if KNOWN_ACKS.match(line):
        return None
    if line.lower() in COLOR_WORDS:
        return ("color", line.lower())

    body = line[5:] if line.startswith("DATA,") else line
    parts = [p.strip() for p in body.split(",")]
    if len(parts) >= 4:
        try:
            r, g, b, c = (int(float(x)) for x in parts[:4])
            return ("color", classify_color(r, g, b, c))
        except ValueError:
            pass
    return ("unknown", line)


# --------------------------------------------------------------------------- #
# Hardware / IO
# --------------------------------------------------------------------------- #
class ScanFilter:
    """Turns the sensor's continuous readings (every 2.5 s) into discrete 'scans'.

    A marker-colored reading counts as a new scan if the color changed, OR the sensor
    left every marker since the last scan (a non-marker reading came in between), OR
    RESCAN_SECS passed. This stops the dot you are still holding the sensor on from
    being counted again as soon as the next instruction is spoken.
    """

    def __init__(self, valid_colors: set[str], rescan_secs: float):
        self.valid, self.rescan = valid_colors, rescan_secs
        self.last_color: Optional[str] = None
        self.last_time = 0.0
        self.released = True

    def feed(self, color: str) -> Optional[str]:
        now = time.time()
        if color not in self.valid:
            self.released = True
            return None
        if (color != self.last_color or self.released
                or now - self.last_time >= self.rescan):
            self.last_color, self.last_time, self.released = color, now, False
            return color
        return None


class ArduinoDevice:
    def __init__(self, port: str, baud: int, events: queue.Queue, debug: bool = False):
        self.debug = debug
        self.unknown_count = 0
        self.last_data: Optional[float] = None
        self.opened_at = time.time()
        if not port:
            raise RuntimeError("ARDUINO_PORT is not set (see .env)")
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("Install pyserial: python3 -m pip install pyserial") from exc
        self.events = events
        self.lock = threading.Lock()
        self.serial = serial.Serial(port, baud, timeout=1)
        print(f"[Serial open: {port} @ {baud}]")
        time.sleep(2)  # Arduino resets when the port opens
        threading.Thread(target=self._read_loop, daemon=True).start()

    def send(self, command: str):
        try:
            with self.lock:
                self.serial.write((command + "\n").encode())
        except Exception as exc:
            print(f"[serial write failed: {exc}]")

    def _read_loop(self):
        while True:
            try:
                line = self.serial.readline().decode("utf-8", errors="ignore")
                if self.debug and line.strip():
                    print(f"[serial] {line.strip()}")
                ev = parse_device_line(line)
                if ev and ev[0] == "unknown":
                    self.unknown_count += 1
                    if self.unknown_count <= 5:
                        print(f"[serial: unrecognized line {ev[1]!r}]")
                    continue
                if ev and ev[0] == "color":
                    self.last_data = time.time()
                if ev:
                    self.events.put(ev)
            except Exception as exc:
                print("[serial read error]", exc)
                time.sleep(1)


class SimDevice:
    """Keyboard stand-in for the wristband + microphone."""

    def __init__(self, building: Building, events: queue.Queue):
        self.b, self.events = building, events
        threading.Thread(target=self._loop, daemon=True).start()

    def send(self, command: str):
        print(f"[wristband <- {command}]")

    def _loop(self):
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"marker\s+(\S+)", line)
            if m and m.group(1) in self.b.nodes:
                color = self.b.nodes[m.group(1)]["marker_color"]
                self.events.put(("color", color))
                self.events.put(("color", "gray"))  # sensor moves off the marker
            else:
                self.events.put(("voice", line))
        self.events.put(("quit", None))


class VoiceListener:
    """Continuously listens for commands; pauses while the app is speaking."""

    def __init__(self, events: queue.Queue, speaking: threading.Event):
        self.events, self.speaking = events, speaking
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        try:
            import speech_recognition as sr
        except ImportError:
            print("[SpeechRecognition/PyAudio missing: voice input disabled]")
            return
        rec = sr.Recognizer()
        rec.pause_threshold = 1.3        # tolerate pauses before the room number
        rec.non_speaking_duration = 0.7
        rec.dynamic_energy_threshold = True
        try:
            mic = sr.Microphone()
            with mic as src:
                rec.adjust_for_ambient_noise(src, duration=1.0)
        except Exception as exc:
            print(f"[microphone unavailable: {exc}]")
            return
        while True:
            if self.speaking.is_set():
                time.sleep(0.1)
                continue
            try:
                with mic as src:
                    audio = rec.listen(src, timeout=6, phrase_time_limit=10)
                text = rec.recognize_google(audio)  # NOTE: audio goes to Google's cloud
                print("Heard:", text)
                self.events.put(("voice", text))
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                continue
            except Exception as exc:
                print(f"[voice error: {exc}]")
                time.sleep(2)


class Buzzer:
    """Audio cues built only from the sketch's BUZZER_ON / BUZZER_OFF commands.
    Each pattern alternates ON, OFF, ON, ... durations in seconds."""

    PATTERNS = {
        "LEFT": [0.15, 0.15, 0.15],
        "RIGHT": [0.15, 0.15, 0.15, 0.15, 0.15],
        "STRAIGHT": [0.6],
        "UTURN": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "ARRIVE": [0.6, 0.2, 0.15, 0.15, 0.15],
    }

    def __init__(self, device):
        self.device = device
        self.gen = 0
        self.lock = threading.Lock()

    def play(self, name: str):
        steps = self.PATTERNS.get(name)
        if not steps:
            return
        with self.lock:
            self.gen += 1
            gen = self.gen
        threading.Thread(target=self._run, args=(steps, gen), daemon=True).start()

    def stop(self):
        with self.lock:
            self.gen += 1
        self.device.send("BUZZER_OFF")

    def _run(self, steps, gen):
        for i, dur in enumerate(steps):
            if gen != self.gen:  # a newer cue replaced this one
                return
            self.device.send("BUZZER_ON" if i % 2 == 0 else "BUZZER_OFF")
            time.sleep(dur)
        if gen == self.gen:
            self.device.send("BUZZER_OFF")


class Speaker:
    """Non-blocking TTS. ElevenLabs if configured, else macOS `say`, else print only."""

    def __init__(self):
        self.q: queue.Queue[str] = queue.Queue()
        self.busy = threading.Event()  # set while speaking or queued
        self._client = None
        self._cache: dict[str, bytes] = {}
        self._warned = False
        self.eleven_ok = bool(ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID)
        print(f"[.env: {ENV_LOADED or 'NOT FOUND'}]")
        if self.eleven_ok:
            print("[Voice: ElevenLabs]")
        else:
            print("[Voice: ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID missing -> "
                  + ("macOS `say` fallback]" if sys.platform == "darwin" else "text only]"))
        threading.Thread(target=self._loop, daemon=True).start()

    def say(self, text: str, interrupt: bool = False):
        print(f"[SAY] {text}")
        if interrupt:
            try:
                while True:
                    self.q.get_nowait()
            except queue.Empty:
                pass
        self.busy.set()
        self.q.put(text)

    def wait(self, timeout: float = 120.0):
        """Block until everything queued has been spoken."""
        end = time.time() + timeout
        time.sleep(0.1)
        while self.busy.is_set() and time.time() < end:
            time.sleep(0.1)

    def _loop(self):
        while True:
            text = self.q.get()
            self.busy.set()
            try:
                self._speak(text)
            except Exception as exc:
                print(f"[TTS error: {exc}]")
            finally:
                if self.q.empty():
                    time.sleep(0.4)  # let room echo die down before the mic resumes
                    if self.q.empty():
                        self.busy.clear()

    def _eleven_audio(self, text: str) -> bytes:
        if text not in self._cache:
            from elevenlabs.client import ElevenLabs
            self._client = self._client or ElevenLabs(api_key=ELEVENLABS_API_KEY)
            audio = self._client.text_to_speech.convert(
                voice_id=ELEVENLABS_VOICE_ID, text=text,
                model_id="eleven_multilingual_v2", output_format="mp3_44100_128")
            self._cache[text] = b"".join(audio)
        return self._cache[text]

    @staticmethod
    def _play_mp3(data: bytes):
        if sys.platform == "darwin":  # afplay ships with macOS, no mpv/ffmpeg needed
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(data)
                path = f.name
            try:
                subprocess.run(["afplay", path], check=True)
            finally:
                os.unlink(path)
        else:
            from elevenlabs.play import play
            play(data)

    def _speak(self, text: str):
        if self.eleven_ok:
            try:
                self._play_mp3(self._eleven_audio(text))
                return
            except Exception as exc:
                print(f"[ElevenLabs FAILED: {type(exc).__name__}: {exc}]")
                if not self._warned:
                    self._warned = True
                    print("  Check: API key + voice ID in .env, internet access, "
                          "`pip install elevenlabs`, and your ElevenLabs credit balance.")
        if sys.platform == "darwin":
            subprocess.run(["say", text], check=False)


class CrowdMap:
    """Reads crowd.json written by crowd_monitor.py: {"zones": {node: {"level": 0-1, "updated": ts}}}"""

    def __init__(self, path: Path, override: Optional[dict[str, float]] = None):
        self.path, self.override = path, override or {}

    def levels(self) -> dict[str, float]:
        if self.override:
            return dict(self.override)
        try:
            with self.path.open() as f:
                zones = json.load(f).get("zones", {})
        except (OSError, ValueError):
            return {}
        now = time.time()
        return {n: float(z.get("level", 0.0)) for n, z in zones.items()
                if now - float(z.get("updated", 0)) <= CROWD_STALE_SECS}


# --------------------------------------------------------------------------- #
# Navigation controller
# --------------------------------------------------------------------------- #
class Navigator:
    def __init__(self, building: Building, speaker: Speaker, buzzer: Buzzer, crowd: CrowdMap,
                 avoid_stairs: bool):
        self.b, self.speaker, self.buzzer, self.crowd = building, speaker, buzzer, crowd
        self.avoid_stairs = avoid_stairs
        self.current: Optional[str] = None
        self.heading: Optional[float] = None   # direction of the last edge walked
        self.goal: Optional[str] = None
        self.pending_goal: Optional[str] = None
        self.route: list[str] = []
        self.idx = 0
        self.last_progress = time.time()
        self.last_crowd_check = 0.0
        self.last_text = ""
        self.filter = ScanFilter(building.marker_colors, RESCAN_SECS)
        self.last_seen_color: Optional[str] = None
        self.last_start_prompt = 0.0

    # -- helpers --
    def say(self, text: str, interrupt: bool = False):
        self.last_text = text
        self.speaker.say(text, interrupt)

    def next_leg(self, crowd: dict[str, float]) -> tuple[str, str]:
        a, to = self.route[self.idx], self.route[self.idx + 1]
        return leg_instruction(self.b, a, to, self.heading, self.route[-1], crowd)

    def announce_leg(self, prefix: str = ""):
        crowd = self.crowd.levels()
        text, haptic = self.next_leg(crowd)
        self.buzzer.play(haptic)
        self.say((prefix + " " + text + " Scan the marker when you get there.").strip(),
                 interrupt=True)
        self.last_progress = time.time()

    def start_prompt(self) -> str:
        s = self.b.start_node
        return (f"hold the sensor on the {self.b.color_of(s)} marker at "
                f"{self.b.label(s)} to begin.")

    # -- goal handling --
    def set_goal(self, goal: str):
        if self.current is None:
            self.pending_goal = goal
            self.say(f"Going to {self.b.label(goal)}. First, " + self.start_prompt(), interrupt=True)
            return
        if self.current == goal:
            self.say(f"You are already at {self.b.label(goal)}.", interrupt=True)
            return
        self.goal, self.pending_goal = goal, None
        self.plan(f"Going to {self.b.label(goal)}.")

    def plan(self, prefix: str):
        crowd = self.crowd.levels()
        try:
            self.route = self.b.plan(self.current, self.goal, crowd, self.avoid_stairs)
        except ValueError as exc:
            self.say(f"Sorry, I can't find a way. {exc}", interrupt=True)
            self.goal = None
            return
        self.idx = 0
        steps = len(self.route) - 1
        meters = round(self.b.route_meters(self.route))
        note = ""
        if any(self.b.nodes[n]["type"] == "staircase" for n in self.route[1:-1]):
            note = " This route passes the staircase."
        self.announce_leg(f"{prefix} {steps} stops, about {meters} meters.{note}")

    # -- events --
    def on_color(self, color: str):
        if color != self.last_seen_color:  # handy when tuning marker colors
            print(f"[sensor sees: {color}]")
            self.last_seen_color = color
        marker = self.filter.feed(color)
        if not marker:
            return

        # Nothing scanned yet: the first scan must be the starting marker (first node).
        if self.current is None:
            start = self.b.start_node
            if marker == self.b.color_of(start):
                self.on_node(start)
            else:
                self.say(f"That is the {marker} marker. Please " + self.start_prompt())
            return

        expected = (self.route[self.idx + 1]
                    if self.goal and self.idx + 1 < len(self.route) else None)
        node = self.b.node_for_marker(marker, self.current, expected)
        if node is None:
            self.say("I saw a marker but can't tell which one. "
                     "If you know where you are, say: I am at, and the room name.")
            return
        print(f"[scan {marker} -> {self.b.label(node)}]")
        self.on_node(node)

    def on_node(self, node: str):
        prev = self.current
        self.current = node
        if prev and prev != node and node in self.b.graph[prev]:
            self.heading = self.b.graph[prev][node]["heading"]
        elif prev != node:
            self.heading = None  # jumped somewhere non-adjacent: facing unknown

        if self.pending_goal:
            self.set_goal(self.pending_goal)
            return
        if not self.goal:
            if prev is None:
                self.say(f"You are at {self.b.label(node)}. Where would you like to go?",
                         interrupt=True)
            elif prev != node:
                self.say(f"You are at {self.b.label(node)}.", interrupt=True)
            return

        self.last_progress = time.time()
        if node == self.goal:
            self.buzzer.play("ARRIVE")
            self.say(f"You have arrived at {self.b.label(node)}. "
                     "Say take me to, and a place, for another trip.", interrupt=True)
            self.goal, self.route = None, []
        elif node in self.route[self.idx + 1:]:
            self.idx = self.route.index(node, self.idx + 1)
            self.announce_leg()
        elif node == self.route[self.idx]:
            self.announce_leg("You are still at " + self.b.label(node) + ".")
        else:
            self.plan(f"You are at {self.b.label(node)}. Recalculating.")

    def on_voice(self, text: str):
        intent, node = parse_utterance(self.b, text)
        if intent == "goto":
            self.set_goal(node)
        elif intent == "here":
            self.on_node(node)
        elif intent == "need_place":
            self.say("Where would you like to go? Say a room number, bathroom, or staircase.",
                     interrupt=True)
        elif intent == "stop":
            self.goal, self.pending_goal, self.route = None, None, []
            self.buzzer.stop()
            self.say("Okay, navigation stopped.", interrupt=True)
        elif intent == "repeat":
            self.say(self.last_text or "Nothing to repeat yet.", interrupt=True)
        elif intent == "where":
            place = self.b.label(self.current) if self.current else None
            self.say(f"You are at {place}." if place else
                     "I don't know yet. Touch a door marker with your wristband.", interrupt=True)
        else:
            print(f"[ignored: {text!r}]")

    def tick(self):
        now = time.time()
        if self.current is None:
            if now - self.last_start_prompt > 30:
                self.last_start_prompt = now
                self.say("Please " + self.start_prompt())
            return
        if not self.goal:
            return
        if now - self.last_crowd_check >= CROWD_CHECK_SECS:
            self.last_crowd_check = now
            crowd = self.crowd.levels()
            try:
                alt = self.b.plan(self.current, self.goal, crowd, self.avoid_stairs)
            except ValueError:
                alt = None
            remaining = self.route[self.idx:]
            if alt and alt != remaining and (
                    self.b.route_cost(alt, crowd, self.avoid_stairs) + 1
                    < self.b.route_cost(remaining, crowd, self.avoid_stairs)):
                self.route, self.idx = alt, 0
                self.announce_leg("Crowd ahead. Taking a different way.")
                return
        if now - self.last_progress > REMIND_SECS:
            self.announce_leg("Still heading to " + self.b.label(self.goal) + ".")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_crowd_overrides(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items or []:
        node, _, val = item.partition("=")
        out[node] = float(val)
    return out


_last_data_warning = 0.0


def check_sensor_data(device):
    """Warn if the Arduino has not sent any DATA lines recently."""
    global _last_data_warning
    if not isinstance(device, ArduinoDevice):
        return
    now = time.time()
    quiet_since = device.last_data or device.opened_at
    if now - quiet_since > 10 and now - _last_data_warning > 15:
        _last_data_warning = now
        print("[WARNING: no DATA lines received from the Arduino for "
              f"{int(now - quiet_since)} s. Run with --debug-serial to see raw lines. "
              "Check: sketch uploaded, Serial Monitor closed, correct ARDUINO_PORT, baud 115200.]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-route", nargs=2, metavar=("START", "GOAL"))
    ap.add_argument("--avoid-stairs", action="store_true")
    ap.add_argument("--crowd", nargs="*", metavar="NODE=LEVEL",
                    help="Override crowd levels (0-1), e.g. bathroom=0.9")
    ap.add_argument("--simulate", action="store_true", help="Keyboard instead of hardware")
    ap.add_argument("--port", default=ARDUINO_PORT)
    ap.add_argument("--speak", action="store_true",
                    help="With --test-route: also read the directions aloud")
    ap.add_argument("--debug-serial", action="store_true",
                    help="Print every raw line received from the Arduino")
    ap.add_argument("--test-voice", action="store_true",
                    help="Speak a sample sentence to check .env / ElevenLabs, then exit")
    args = ap.parse_args()

    building = Building(BUILDING_FILE)
    crowd = CrowdMap(CROWD_FILE, parse_crowd_overrides(args.crowd))

    shared = building.shared_markers()
    if shared:
        print(f"[Shared marker colors {shared}: nodes are told apart by the guided order.]")
    print(f"[Start node: {building.label(building.start_node)}]")
    if not building.calibrated:
        print("WARNING: building.json has placeholder distances/headings. "
              "Left/right instructions will not match the real hallways until calibrated.")

    if args.test_voice:
        sp = Speaker()
        sp.say("Voice check. If you can hear this, the wristband navigator can speak.")
        sp.wait()
        return

    if args.test_route:
        start, goal = args.test_route
        levels = crowd.levels()
        route = building.plan(start, goal, levels, args.avoid_stairs)
        steps = describe_route(building, route, None, levels)
        print("Route:", " -> ".join(building.label(n) for n in route))
        if args.speak:
            sp = Speaker()
            for step in steps:
                sp.say(step)
            sp.wait()
        else:
            print("\n".join(steps))
        return

    events: queue.Queue = queue.Queue()
    speaker = Speaker()
    if args.simulate:
        device = SimDevice(building, events)
    else:
        try:
            device = ArduinoDevice(args.port, ARDUINO_BAUD, events, args.debug_serial)
        except Exception as exc:
            print("Cannot connect to Arduino:", exc)
            sys.exit(1)
        VoiceListener(events, speaker.busy)

    nav = Navigator(building, speaker, Buzzer(device), crowd, args.avoid_stairs)
    nav.last_start_prompt = time.time()
    speaker.say("Wristband ready. Please " + nav.start_prompt())

    while True:
        try:
            kind, value = events.get(timeout=1.0)
        except queue.Empty:
            nav.tick()
            check_sensor_data(device)
            continue
        except KeyboardInterrupt:
            break
        try:
            if kind == "color":
                nav.on_color(value)
            elif kind == "voice":
                nav.on_voice(value)
            elif kind in ("error", "info"):
                print("[device]", value)
            elif kind == "quit":
                time.sleep(1)  # let queued speech finish in simulate mode
                break
            nav.tick()
            check_sensor_data(device)
        except Exception as exc:  # never crash while someone is walking
            print("Navigation error:", exc)
    speaker.wait(10)
    print("Exiting.")


if __name__ == "__main__":
    main()