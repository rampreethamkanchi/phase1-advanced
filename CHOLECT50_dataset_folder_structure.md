# Structure of CHOLECT50 dataset at /raid/manoranjan/rampreetham/CholecT50
# Note: CholecT50 is a superset of CholecT45, containing 50 videos in total.

.
├── label_mapping.txt       # Mapping of triplet IDs to component IDs (Instrument, Verb, Target)
├── labels/                 # Annotation files in JSON format for each video
│   ├── VID01.json
│   ├── VID02.json
│   ├── VID04.json
│   ├── VID05.json
│   ├── VID06.json
│   ├── VID08.json
│   ├── VID10.json
│   ├── VID12.json
│   ├── VID13.json
│   ├── VID14.json
│   ├── VID15.json
│   ├── VID18.json
│   ├── VID22.json
│   ├── VID23.json
│   ├── VID25.json
│   ├── VID26.json
│   ├── VID27.json
│   ├── VID29.json
│   ├── VID31.json
│   ├── VID32.json
│   ├── VID35.json
│   ├── VID36.json
│   ├── VID40.json
│   ├── VID42.json
│   ├── VID43.json
│   ├── VID47.json
│   ├── VID48.json
│   ├── VID49.json
│   ├── VID50.json
│   ├── VID51.json
│   ├── VID52.json
│   ├── VID56.json
│   ├── VID57.json
│   ├── VID60.json
│   ├── VID62.json
│   ├── VID65.json
│   ├── VID66.json
│   ├── VID68.json
│   ├── VID70.json
│   ├── VID73.json
│   ├── VID74.json
│   ├── VID75.json
│   ├── VID78.json
│   ├── VID79.json
│   ├── VID80.json
│   ├── VID92.json
│   ├── VID96.json
│   ├── VID103.json
│   ├── VID110.json
│   └── VID111.json
├── LICENSE
├── README.md
└── videos/                 # Video folders containing frame images (png)
    ├── VID01
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    ├── VID02
    ├── VID04
    ├── VID05
    ├── VID06
    ├── VID08
    ├── VID10
    ├── VID12
    ├── VID13
    ├── VID14
    ├── VID15
    ├── VID18
    ├── VID22
    ├── VID23
    ├── VID25
    ├── VID26
    ├── VID27
    ├── VID29
    ├── VID31
    ├── VID32
    ├── VID35
    ├── VID36
    ├── VID40
    ├── VID42
    ├── VID43
    ├── VID47
    ├── VID48
    ├── VID49
    ├── VID50
    ├── VID51
    ├── VID52
    ├── VID56
    ├── VID57
    ├── VID60
    ├── VID62
    ├── VID65
    ├── VID66
    ├── VID68
    ├── VID70
    ├── VID73
    ├── VID74
    ├── VID75
    ├── VID78
    ├── VID79
    ├── VID80
    ├── VID92
    ├── VID96
    ├── VID103
    ├── VID110
    └── VID111

# Dataset Note
CholecT50 provides labels for:
- triplets: <instrument, verb, target>
- instruments
- verbs/actions
- targets/anatomies
- phases (Surgical phase recognition)

Each JSON file in the `labels/` directory contains annotations for every frame of the corresponding video.
The user's primary interest in this dataset is for **phase detection**.
