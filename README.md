# Mandarin Visual Speech Representation Tool

Tool to help visualize speech for the purpose of learning Mandarin. The initial intent behind this project is that I personally have not found a platform that would visual speech for Mandarin learners. Visuallizing speech can help learners understand and see the subtle changes in pitch for a tonal language like Mandarin.

## Prerequisites

Make sure you have Python 3.6 or later installed.

## Installation

Follow the instructions based on your operating system:

### For **Windows** Users:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/kid-codei/mandarin_visual_speech_representation_tool.git
    cd mandarin_visual_speech_representation_tool
    ```

2. **Set up a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the app:**
    ```bash
    python app.py
    ```

### For **Mac/Linux** Users:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/kid-codei/mandarin_visual_speech_representation_tool.git
    cd mandarin_visual_speech_representation_tool
    ```

2. **Set up a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the app:**
    ```bash
    source venv/bin/activate     
    python app.py
    ```

### To Set Up the Project Locally:

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/mandarin_visual_speech_representation_tool.git
    ```

2. Navigate into the project directory:
    ```bash
    cd mandarin_visual_speech_representation_tool
    ```

3. Set up the virtual environment and install dependencies (follow the OS-specific instructions above).

4. Run the application with:
    ```bash
    python app.py
    ```


```
mandarin_visual_speech_representation_tool/
├── db/
│   └── sentences.db          # SQLite database storing sentence data
├── data/
│   ├── sentence_data.json    # JSON with sentence information
│   └── features.json         # JSON that will hold pitch and alignment information for sentences
├── scripts/
│   ├── _archive
│   ├── batch_text_pitch_alignment.py # Script to align the segmented words to the pitches in those locations
│   ├── create_database.py    # Creates SQLite DB and table
│   ├── export_sentences_json.py # Export the sentence data in a JSON file
│   ├── import_sentence_from_json.py # Inserts initial sentence/audio data into database
│   ├── pitch_alignment.py    # Script to align the audio to the 
│   ├── plot_pitch_contour.py # Script to plot the pitch contour of the sentences
│   ├── process_audio.py      # Get the pitch values for the sample sentences
│   ├── test_db.py            # Sample query to test database functionality
│   ├── update_chinese_variants.py # Add simplified text and pinyin to the database
│   └── update_features.py    # Update database with pitch and alignment information
├── static/
│   ├── audio/                # Audio samples to be used in this tool
│   │   ├── sample1.wav
│   │   ├── sample2.wav
│   │   └── ...
│   ├── json/                 # JSON files for future configs
│   │   ├── align             # JSON with the alignment information for the sentencse
│   │   ├── aligned_pitch     # JSON with the sentence alignment alongside what the corresponding pitches are
│   │   ├── pitch             # JSON with the pitch values of the sentences corresponding audio
│   │   ├── plots.            # Images of the plotted pitch curves of sentences (handled dynamically in current code)
│   └── learner_audios/       # Folder where learner recordings will be kept
├── templates/
│   ├── index.html            # HTML for the front-end website
├── app.py                    # Flask app for the back-end 
├── README.md                 # Instructures for installation
└── requirements.txt          # Python dependencies
```

### Next Steps
- Standardizing pitch between male/female speakers (high or low vocal ranges)
- Improvements to segmentation accuracy
- Color coding words
- Grouping characters that are considered 1 word
- Facelift on the UI

Please feel free to reach out to me with questions. I will be working on this on the side in the near future.