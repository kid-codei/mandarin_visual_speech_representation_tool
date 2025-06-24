# mandarin_visual_speech_representation_tool
Tool to help visualize speech for the purpose of learning Mandarin. CS 6460 Education Technology Project.

mandarin_visual_speech_representation_tool/
├── db/
│   └── sentences.db          # SQLite database storing sentence data
├── data/
│   └── sentence_data.json    # JSON with sentence information
│   └── features.json         # JSON that will hold pitch and alignment information for sentences
├── scripts/
│   ├── create_database.py    # Creates SQLite DB and table
│   ├── insert_sentence_data.py # Inserts initial sentence/audio data into database
│   ├── update_features.py    # Update databse with pitch and alignment information
│   └── test_db.py            # Sample query to test database functionality
├── static/
│   ├── audio/                # Audio samples to be used in this tool
│   │   ├── sample1.wav
│   │   ├── sample2.wav
│   │   └── ...
│   ├── js/                   # Javascript files for future interactivity
│   ├── json/                 # JSON files for future configs
│   └── learner_audios/       # Folder where learner recordings will be kept
├── README.md
└── requirements.txt          # Python dependencies

