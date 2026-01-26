# Pipeline de diarização e extração de locutor (CPU/GPU)

Este repositório contém um fluxo simples para:
1) **Diarização** (identificar quem fala e quando)
2) **Extração** dos trechos de um locutor específico no áudio original

Os scripts foram projetados para **rodar em CPU** e também **acelerar em GPU** (RTX 3050).

---

## 1) Pré‑requisitos

### 1.1 Python e venv
Crie/ative o ambiente virtual:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 1.2 Dependências principais
Instale os pacotes:

```powershell
pip install pyannote.audio speechbrain torchaudio soundfile huggingface_hub
```

> **Observação:** `soundfile` é usado para ler o WAV sem depender do `torchcodec`.

### 1.3 Token do Hugging Face
Você precisa aceitar os termos dos modelos:
- `pyannote/speaker-diarization`
- `pyannote/speaker-diarization-community-1`

Depois exporte o token:

```powershell
$env:HF_TOKEN="hf_...seu_token..."
```

---

## 2) (Opcional) GPU – RTX 3050

Se quiser acelerar com GPU:

1) Instale o PyTorch com CUDA (use a versão recomendada no site do PyTorch).
2) Verifique se a GPU está visível:
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Se retornar `True`, a GPU será usada automaticamente.

---

## 3) Configuração dos scripts

### 3.1 diarize.py
No topo do arquivo:
- `ORIGINAL_AUDIO_FILE` → nome do áudio original
- `DIARIZE_AUDIO_FILE` → versão 16 kHz mono (gerada automaticamente)
- `DIARIZE_FORCE_CONVERT` → `True` se quiser reconverter o áudio
- `DIARIZE_TEST_SECONDS` (via env) → limita para teste rápido

Variáveis de ambiente úteis:
- `DIARIZE_USE_GPU=1` (padrão)
- `DIARIZE_SEG_BATCH=32`
- `DIARIZE_EMB_BATCH=64`

### 3.2 extract_dublador.py
No topo do arquivo:
- `TARGET_SPEAKER` → label do locutor (ex: `SPEAKER_02`)
- `MIN_SEGMENT_MS` → corte mínimo (ex: 300 ms)
- `OUTPUT_WAV` → arquivo final

---

## 4) Rodar diarização

### 4.1 Teste rápido (30s)
```powershell
$env:DIARIZE_TEST_SECONDS="30"
.\venv\Scripts\python.exe .\diarize.py
```

Se terminar, o processo está OK.

### 4.2 Execução completa
```powershell
Remove-Item Env:DIARIZE_TEST_SECONDS -ErrorAction SilentlyContinue
.\venv\Scripts\python.exe .\diarize.py
```

Saída esperada:
- `speakers.txt` com timestamps e labels

---

## 5) Identificar o locutor alvo

Abra o `speakers.txt` e identifique o speaker correto (ex: `SPEAKER_02`).

---

## 6) Extrair o dublador

Edite `TARGET_SPEAKER` em `extract_dublador.py` e rode:

```powershell
.\venv\Scripts\python.exe .\extract_dublador.py
```

Saída:
- `jarvis_dublador_raw.wav` (qualidade do áudio original)

---

## 7) Observações importantes

- A diarização usa **16 kHz mono** para ser mais rápida.
- A extração final é feita **no áudio original**, preservando qualidade.
- Em CPU, 2h de áudio pode levar muitas horas.
- Em GPU, o tempo reduz bastante, mas o clustering continua no CPU.

---

## 8) Dicas para acelerar

- Use GPU quando possível.
- Ajuste `DIARIZE_SEG_BATCH` / `DIARIZE_EMB_BATCH` (reduza se der OOM).
- Faça diarização por blocos se necessário.

---

Se quiser mais automações (exportar amostras por speaker, logs de progresso, etc.), me avise.
