# Docker — Deployment dell'inferenza GUV

Note di lavoro sulla dockerizzazione di `guv_detection/tools/inference.py`, per garantire che l'inferenza si comporti allo stesso modo su Windows e Linux, senza dipendere dall'ambiente conda locale di ogni macchina.

## Perché questo esiste

L'app desktop (`guv_detection/app/`) resta distribuita via PyInstaller (`.exe`) — è la via giusta per un'interfaccia grafica consegnata a un utente non tecnico, ma richiede un build separato per ogni sistema operativo e non garantisce che l'ambiente sia identico tra macchine diverse.

Questo setup Docker copre invece il caso "CLI headless": inferenza + analisi dimensionale (`tools/inference.py`), pensato per essere **riproducibile byte-per-byte a livello di versioni di dipendenze** su qualunque macchina con Docker installato (Windows, Linux, in futuro anche edge device tipo Jetson).

## Decisioni prese (e perché)

- **Base image**: `python:3.10-slim`, non `ultralytics/ultralytics:latest-cpu`. L'immagine ufficiale di Ultralytics porta versioni scelte da loro, non necessariamente identiche a quelle del conda env `guv` usato in sviluppo. Si parte da un'immagine Python "vuota" e si installano esattamente le versioni pinnate.
- **`requirements.txt` pinnato dal `pip freeze` reale del conda env `guv`** (Python 3.10.19), non versioni generiche. Escluse solo le dipendenze Windows-only di PyInstaller (`pefile`, `pywin32-ctypes`, con marker `; sys_platform == "win32"` così lo stesso file resta valido anche su Linux/Docker).
- **`torch`/`torchvision` installati dalla build CPU-only** (`--extra-index-url https://download.pytorch.org/whl/cpu`): sul Windows locale torch include CUDA (c'è la GPU), ma il container CPU non ha GPU — stessa versione `2.9.0`, binario diverso ma comportamento/API equivalenti.
- **`libgl1` + `libglib2.0-0` installati via `apt`**: `python:3.10-slim` non ha librerie grafiche di sistema; `opencv-python` (dipendenza di `ultralytics`) le richiede anche solo per essere importato, anche se non le usiamo mai (niente GUI in un container). Senza, crash `ImportError: libGL.so.1`.
- **`Dockerfile.cpu` dentro `docker/`, build context = root del repo**: il file sta in una sottocartella per tenere il repo ordinato, ma il *build context* resta la root, perché `COPY` non può leggere file fuori dal context (quindi non si può spostare anche `guv_detection/` dentro `docker/`). Si builda sempre da root con `-f docker/Dockerfile.cpu`.
- **`docker/Dockerfile.cpu.dockerignore`**: Docker/BuildKit riconosce l'ignore-file specifico di un Dockerfile solo se si chiama esattamente `<nome-dockerfile>.dockerignore` — va rinominato ogni volta che si rinomina il Dockerfile stesso.
- **Pesi del modello (`best.pt` grey/rgb) copiati dentro l'immagine**: sono piccoli (modello nano), ha senso includerli così l'immagine è autosufficiente e serve montare solo la cartella delle immagini da processare, non anche i pesi.

## File in questa cartella

- `Dockerfile.cpu` — ricetta di build (immagine CPU-only, per inferenza)
- `Dockerfile.cpu.dockerignore` — esclusioni dal build context (`.git`, dataset, notebook, immagini di repo, ecc. — tiene il context sotto i ~11 MB invece di ~930 MB)

## Comandi

Build (da root del repo):
```bash
docker build -t guv-infer -f docker/Dockerfile.cpu .
```

Run (Windows — path del volume con lettera di drive):
```powershell
docker run --rm -v "C:\path\alle\immagini:/data" guv-infer --model guv_detection/app/model/rgb/yolo11_n/best.pt --folder /data --mu_per_pixel 0.339 --conf_thresh 0.25
```

Run (Linux — path normale, niente lettera di drive; niente Docker Desktop, il demone gira come servizio nativo):
```bash
docker run --rm -v /home/utente/path/alle/immagini:/data guv-infer --model guv_detection/app/model/rgb/yolo11_n/best.pt --folder /data --mu_per_pixel 0.339 --conf_thresh 0.25
```

Altri comandi utili:
```bash
docker images          # elenca le immagini locali
docker rmi guv-infer    # cancella l'immagine (aggiungi -f se serve forzare)
```

Note sui parametri di `docker run`:
- `--rm` → il container viene eliminato subito dopo l'esecuzione (l'immagine resta)
- `-v host:/data` → monta la cartella host con le immagini reali dentro il container come `/data`; i risultati (`predict/`, CSV, PDF) vengono scritti lì e restano sul disco host dopo che il container termina
- `--model` è relativo a `WORKDIR /app` dentro il container (dove i pesi sono stati copiati in fase di build)

## Stato attuale / prossimi passi

- [x] Build funzionante su Windows (risolto crash `libGL.so.1`)
- [x] Run di test completato su Windows
- [ ] Validare build + run identici su Linux, stessa immagine costruita dallo stesso `Dockerfile.cpu` — è il vero test di "funziona uguale ovunque"
- [ ] Da ricordare: `docker/`, `requirements.txt` e `guv_detection/tools/inference.py` aggiornati vanno committati e pushati **prima** di clonare su Linux, altrimenti la macchina Linux non ha nulla di questo lavoro
- [ ] Possibile step futuro: variante GPU (`Dockerfile.gpu`) per il training, e/o deployment su edge device (Jetson) — richiede immagini base NVIDIA L4T specifiche per la versione di JetPack installata, diverse da queste CPU-only (vedi discussione in conversazione: architettura ARM64 vs x86_64, build nativa sul device)

## Modifiche fatte a `inference.py` per questo lavoro

- Rimosso `plt.show()` e forzato il backend matplotlib `Agg` (`matplotlib.use('Agg')`) — necessario perché un container non ha display
- Allineata la logica di analisi a quella di `app.py`: fit log-normale (`scipy.stats.lognorm`), istogramma con `density=True`, export CSV (`GUV_ID`, `Diameter_um`) oltre al PDF — prima mancavano fit e CSV
