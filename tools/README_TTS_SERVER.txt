# Jarvis TTS setup notes
# 1) Install dependencies:
#    pip install fastapi uvicorn pydantic
# 2) Run server:
#    uvicorn tools.tts_server:app --host 0.0.0.0 --port 8000
# 3) Configure backend commands:
#    set env vars:
#      GPT_SOVITS_CMD, FISH_CMD, F5_CMD, TTS_REF_WAV
#    Each command should output a wav at {out}.
