# TT-Boltz Quad-Card Live Demo

Zero-input demo that runs Boltz-2 on every Tenstorrent card in the system at
once, looping through 16 example complexes (single proteins, protein–ligand,
protein–protein) and streaming live diffusion frames into four Mol* viewers
side-by-side. Ligands render as element-coloured ball-and-stick inside the
protein cartoon so co-crystal partners (heme, indinavir, biotin, …) are
actually visible.

The point is visual: show every card kept busy, with structures continuously
taking shape on screen. User controls are intentionally minimal — admin-only
Start / Stop links in the footer.

The demo is designed to **run for days** unattended: workers are supervised
and respawned on crash, log files rotate, parent death kills children via
`PR_SET_PDEATHSIG`, SSE clients can drop and reconnect freely without
disturbing the prediction loop.

## The rotation

16 complexes, every one ≤ 768 protein residues total. Sequences come
straight from the canonical RCSB FASTA download for each PDB ID — no
hand-typing, no species substitutions. Ligand CCD codes verified against
each wwPDB nonpolymer entity record.

| #  | name                       | seq_len | kind            | PDB                       |
|----|----------------------------|--------:|-----------------|---------------------------|
| 0  | HIV protease · indinavir   |     198 | protein–ligand  | 1HSG (homodimer + MK1)    |
| 1  | myoglobin · heme           |     153 | protein–ligand  | 1MBN (+ HEM)              |
| 2  | insulin                    |      51 | protein         | 4INS (A + B chains)       |
| 3  | hemoglobin α₂β₂           |     574 | protein         | 1A3N (α₂β₂ tetramer)     |
| 4  | ubiquitin                  |      76 | protein         | 1UBQ                      |
| 5  | trypsin · benzamidine      |     223 | protein–ligand  | 3PTB (+ BEN)              |
| 6  | barnase · barstar          |     199 | protein–protein | 1BRS (enzyme + inhibitor) |
| 7  | lysozyme                   |     129 | protein         | 1HEL                      |
| 8  | CA-I · acetazolamide       |     260 | protein–ligand  | 1AZM (+ AZM)              |
| 9  | MDM2 · p53 peptide         |     124 | protein–protein | 1YCR (oncology peptide)   |
| 10 | cytochrome c               |     104 | protein         | 1HRC                      |
| 11 | DHFR · methotrexate        |     159 | protein–ligand  | 4DFR (+ MTX)              |
| 12 | calmodulin · MLCK peptide  |     174 | protein–protein | 2BBM                      |
| 13 | crambin                    |      46 | protein         | 1CRN                      |
| 14 | streptavidin · biotin      |     159 | protein–ligand  | 1STP (+ BTN)              |
| 15 | trypsin · BPTI             |     281 | protein–protein | 2PTC (enzyme + inhibitor) |

**Distribution**: 6 protein, 6 protein–ligand, 4 protein–protein.

`protein` covers both single chains *and* oligomeric proteins whose
biological unit happens to have multiple chains (insulin A+B, hemoglobin
α₂β₂). `protein–protein` is reserved for *distinct* proteins
interacting with each other.

The first four entries open the demo with a deliberately diverse tableau
(drug-target complex, cofactor binder, two-chain hormone, full tetramer)
since `processor N` always gets `complex N` in the first round.

To change the rotation, edit [`demo_quad/complexes.py`](complexes.py) and
restart the demo. Three small builder primitives (`_Chain`, `_Ligand`,
`_build`) compose any combination of chains, homo-multimers, and ligands;
the file asserts `len == 16` and `seq_len ≤ MAX_SEQ_LEN` at import time so
accidental violations fail loudly.

## Work dispatch

A single shared atomic counter hands complexes out to workers. The very
first round is deterministic — card 0 takes complex 0, card 1 takes
complex 1, … — so a fresh demo always opens with the same four panels.
From there it's work-stealing: whichever card finishes first grabs the
next index (4, 5, 6, …). No card sits idle waiting on a slower one, the
rotation wraps modulo 16, and each unique sequence hits ColabFold for
its MSA exactly once across all four workers because the on-disk MSA
cache is shared.

## Run

```bash
pip install -e .                          # gives you tt-boltz + the demo
tt-boltz-demo-quad --autoplay             # binds 0.0.0.0:5001 by default
# or
python -m demo_quad.app --autoplay
```

Open `http://<this-machine>:5001/` from a browser (over Tailscale: the host
short name or `100.*` IP works). Click **Start** in the footer if you didn't
pass `--autoplay`.

Recommended long-running setup: run under `systemd` (a sample unit lives in
the comments at the end of this README) so the OS restarts the demo on
panic, log rotation is centralized, and graceful stop is just
`systemctl stop`.

## Architecture

```
                            Browser
                               │
                               │ SSE  (one connection, 4 panels)
                               ▼
   ┌────────────────────── Flask app (main process) ──────────────────────┐
   │                                                                      │
   │   ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐    │
   │   │ HTTP routes  │ ◄──│ SSE fan-out      │ ◄──│ Drain thread    │    │
   │   │ /, /events,  │    │ (per-client      │    │ (sole consumer  │    │
   │   │ /start /stop │    │  bounded queue)  │    │  of mp.Queue)   │    │
   │   └──────────────┘    └──────────────────┘    └────────▲────────┘    │
   │                                                        │             │
   │   ┌─────────────────────────────────────────────┐      │ events      │
   │   │ Supervisor thread                           │      │             │
   │   │   • polls process.is_alive() every 2 s      │      │             │
   │   │   • respawns crashed workers (backoff)      │      │             │
   │   │   • surfaces failures as 'error' events     │      │             │
   │   └─────────┬───────────────────────────────────┘      │             │
   │             │ spawn / sigterm                          │             │
   └─────────────┼──────────────────────────────────────────┼─────────────┘
                 │                                          │
                 ▼                                          │
   ┌─────────────────────────────────────────────┐          │
   │ Worker subprocess (one per TT card)         │          │
   │   • TT_VISIBLE_DEVICES set before any ttnn  │ ─────────┘
   │   • PR_SET_PDEATHSIG so parent death kills  │
   │   • logs to ~/.boltz/logs/quad-worker-N.log │
   │   • model + features loaded once            │
   │   • predict_step loop gated by play event   │
   └─────────────────────────────────────────────┘
```

### Module layout

| file              | responsibility |
|-------------------|---------------|
| `app.py`          | CLI + Flask routes + SSE handler. Thin wrapper around `pool.py`. |
| `pool.py`         | `CardPool`: owns subprocesses, drain thread, supervisor, SSE subscribers, shared dispatch counter. |
| `worker.py`       | Subprocess entrypoint. Loads model, pulls next complex from the dispatcher, runs predict loop, emits events. |
| `complexes.py`    | The 16-entry rotation (Boltz YAML per entry). |
| `templates/index.html` | UI: 2×2 Mol* grid with a `cover/banner` state machine. |

### Failure handling

| failure                          | what happens |
|----------------------------------|--------------|
| Single worker crashes mid-predict | Supervisor detects within 2 s, emits an `error` event so its panel shows "Error", then respawns the worker with exponential backoff (2 → 4 → 8 → … capped at 30 s). |
| Main Flask process killed         | `PR_SET_PDEATHSIG=SIGKILL` in each worker → kernel kills all workers immediately. No orphan ttnn subprocesses. |
| SSE client browser closes         | Per-client queue dropped, prediction loop unaffected. |
| SSE client falls behind           | Per-client queue saturates → client is evicted; browser auto-reconnects 1.5 s later and gets a fresh snapshot. |
| Worker startup fails (e.g. TT busy) | Worker exits non-zero; supervisor reports the error and retries with backoff. |
| Reverse proxy idle timeout        | SSE heartbeat (` : ping`) every 15 s keeps the connection live. |
| `SIGINT` / `SIGTERM` to demo      | Workers get SIGTERM, 15 s grace, then SIGKILL for stragglers. Drain thread flushes tail events before exit. |

### Bandwidth tricks (ported from `demo/`)

The first diffusion frame of every prediction ships a full CIF template
(~50 KB). Subsequent intermediate frames ship only a flat array of atom
coordinates (~10 KB); the browser merges them into the cached template.
The final frame ships in full so the pLDDT-coloured cartoon can render
with proper per-residue confidence.

A 400 ms per-panel render throttle keeps Mol* from queueing up stale
frames when intermediates arrive faster than the GPU can re-tessellate
cartoons.

## Flags

```
tt-boltz-demo-quad
    [--host HOST]                # default 0.0.0.0
    [--port PORT]                # default 5001
    [--autoplay]                 # begin predicting at boot
    [--no-fast]                  # disable block-fp8 fast mode
    [--sampling-steps N]         # diffusion steps per prediction (default 200)
    [--linger-seconds S]         # show final structure between predictions (default 2.0)
    [--log-dir DIR]              # worker log directory (default ~/.boltz/logs)
    [-v|--verbose]               # debug-level main-process logging
```

## Endpoints

| method | path        | purpose |
|--------|-------------|---------|
| GET    | `/`         | Single-page UI |
| GET    | `/events`   | SSE stream of card events (gzip negotiated) |
| GET    | `/status`   | JSON snapshot of pool state |
| POST   | `/start`    | Resume the prediction loop |
| POST   | `/stop`     | Pause workers (model stays resident) |

## Logs

Each worker writes to `~/.boltz/logs/quad-worker-<device>.log`, rotated at
5 MB × 3 backups. The Flask process logs supervisor / drain events to stderr.

```bash
tail -F ~/.boltz/logs/quad-worker-*.log
```

## Configuring the rotation

Edit `demo_quad/complexes.py`. Compose entries from three primitives —
`_Chain(seq, copies=N)` for protein chains (including homo-multimers),
`_Ligand(ccd, copies=N)` for small molecules, and `_build(name, *parts)`
to assemble:

```python
ROTATION: list[Complex] = [
    _build("HIV protease · indinavir",
           _Chain(HIV_PROTEASE, copies=2), _Ligand("MK1")),
    _build("myoglobin · heme",
           _Chain(MYOGLOBIN), _Ligand("HEM")),
    _build("hemoglobin α₂β₂",
           _Chain(HBA, copies=2), _Chain(HBB, copies=2)),
    _build("insulin",
           _Chain(INSULIN_A), _Chain(INSULIN_B)),
    # ...
]
```

For anything more elaborate (pocket constraints, custom MSA files,
SMILES ligands, cyclic peptides, …) drop the full YAML straight into
`Complex(name=..., yaml=..., seq_len=...)`. The module asserts
`len(ROTATION) == 16` and `seq_len <= MAX_SEQ_LEN` at import time so
accidental violations fail loudly.

## Optional: run under systemd

```
# /etc/systemd/system/tt-boltz-demo-quad.service
[Unit]
Description=TT-Boltz quad-card live demo
After=network.target

[Service]
User=ttuser
WorkingDirectory=/home/ttuser/tt-boltz
ExecStart=/home/ttuser/tt-boltz/env/bin/tt-boltz-demo-quad --autoplay
Restart=always
RestartSec=10
KillSignal=SIGTERM
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

Then `sudo systemctl enable --now tt-boltz-demo-quad`. Restarts are handled
by systemd, the supervisor handles per-card crashes inside the process.
