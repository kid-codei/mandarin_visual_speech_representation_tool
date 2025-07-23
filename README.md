# mandarin_visual_speech_representation_tool
Tool to help visualize speech for the purpose of learning Mandarin. CS 6460 Education Technology Project.

mandarin_visual_speech_representation_tool/
├── db/
│   └── sentences.db          # SQLite database storing sentence data
├── data/
│   ├── sentence_data.json    # JSON with sentence information
│   └── features.json         # JSON that will hold pitch and alignment information for sentences
├── scripts/
│   ├── _archive
│   ├── batch_text_pitch_alignment.py
│   ├── create_database.py    # Creates SQLite DB and table
│   ├── export_sentences_json.py
│   ├── import_sentence_from_json.py # Inserts initial sentence/audio data into database
│   ├── pitch_alignment.py
│   ├── plot_pitch_contour.py
│   ├── process_audio.py
│   ├── setup_mac.sh
│   ├── setup_windows.sh
│   ├── test_db.py            # Sample query to test database functionality
│   ├── update_chinese_variants.py
│   └── update_features.py    # Update database with pitch and alignment information
├── static/
│   ├── audio/                # Audio samples to be used in this tool
│   │   ├── sample1.wav
│   │   ├── sample2.wav
│   │   └── ...
│   ├── js/                   # Javascript files for future interactivity
│   ├── json/                 # JSON files for future configs
│   │   ├── align
│   │   ├── aligned_pitch
│   │   ├── pitch
│   │   ├── plots
│   └── learner_audios/       # Folder where learner recordings will be kept
├── templates/
│   ├── index.html            #
├── venv/
│   ├── bin
│   ├── include
│   ├── lib
│   ├── share
│   └── pvenv.cfg
├── app.py
├── README.md
└── requirements.txt          # Python dependencies

