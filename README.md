# Pipeline de diarizacao e extracao de locutor (CPU/GPU)

Este repositorio cria um dataset limpo de um locutor alvo (dublador Eduardo Borgerth) a partir de um podcast longo.
O foco e **limpeza maxima** para clonagem de voz (TTS), mesmo que reduza a quantidade.

---

## 1) Requisitos

### 1.1 Python e venv
```powershell
python -m venv venv
.env\Scripts\Activate.ps1
```

### 1.2 Dependencias
```powershell
pip install -r requirements.txt
```

### 1.3 Tokens (.env)
Crie um arquivo `.env` com:
```
HF_TOKEN=hf_...seu_token...
PYANNOTE_API_KEY=...sua_api_key...
```

---

## 2) Fluxo recomendado (melhor qualidade)

### 2.1 Precision-2 (diarizacao de alta qualidade)
Rodar em modo duplo (exclusive + non-exclusive):
```powershell
$env:RUN_BOTH_EXCLUSIVE="1"
.env\Scripts\python.exe .\identify_precision2.py
```
Saidas:
- `identify_output_non_exclusive.json`
- `identify_output_exclusive.json`
- `speakers_non_exclusive.txt`
- `speakers_exclusive.txt`
- `speakers.txt`

### 2.2 Overlap detection
```powershell
.env\Scripts\python.exe .\detect_overlap.py
```
O script tenta nesta ordem:
1) `identify_output_non_exclusive.json`
2) `identify_output.json`
3) modelo `pyannote/overlapped-speech-detection`
4) `speakers.txt` (fallback)

Opcional (forcar modelo primeiro):
```powershell
$env:PREFER_OVERLAP_MODEL="1"
.env\Scripts\python.exe .\detect_overlap.py
```

### 2.3 Extrair segmentos do dublador
```powershell
.env\Scripts\python.exe .\extract_dublador.py
```
Saidas:
- `jarvis_dublador_raw.wav`
- `segments/` (trechos individuais)
- `segments/review_list.csv`

### 2.4 Revisao manual
Abra `segments/review_list.csv` e marque `approved=1` apenas nos trechos 100% limpos.

### 2.5 Concatenar apenas os aprovados
```powershell
.env\Scripts\python.exe .\concat_approved.py
```
Saida:
- `jarvis_dublador_clean.wav`

---

## 3) Fluxo local (sem Precision-2)
Se nao quiser usar a API:
```powershell
.env\Scripts\python.exe .\diarize.py
```
Depois siga os passos a partir do overlap/extracao.

---

## 4) Configuracoes importantes

### extract_dublador.py
- `TARGET_SPEAKER`
- `MIN_SEGMENT_SECONDS`
- `EDGE_TRIM_SECONDS`
- `SIMILARITY_THRESHOLD`
- `OVERLAP_MAX_RATIO`

### detect_overlap.py
- `OVERLAP_DILATION_SECONDS`
- `MIN_OVERLAP_SECONDS`
- Flags:
  - `PREFER_OVERLAP_MODEL=1`
  - `REQUIRE_OVERLAP_MODEL=1`

### identify_precision2.py
- `RUN_BOTH_EXCLUSIVE=1` (gera exclusive + non-exclusive)
- `MATCHING_THRESHOLD`

---

## 5) Observacoes
- Em Windows, o "Controlled Folder Access" pode bloquear escrita no Desktop. Se ocorrer, rode como Admin ou mova o projeto para `C:\work`.
- Avisos de `torchaudio` sao esperados em alguns ambientes e nao impedem a execucao.
- O dataset final deve ser **100% limpo** para evitar contaminacao da voz clonada.
