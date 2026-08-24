"""
Create the .exe GUV Detector
"""
import PyInstaller.__main__
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

PyInstaller.__main__.run([
    'app_modern.py',                    # Script principale
    '--name=VesiScope',                 # Nome dell'applicazione
    '--onefile',                        # Crea un singolo file eseguibile
    '--windowed',                       # Non mostrare console (solo GUI)
    '--icon=guv_ico.ico',               # Icona (opzionale, crea un file icon.ico)
    '--add-data=model:model',           # Includi la cartella model
    '--hidden-import=PIL._tkinter_finder',
    '--hidden-import=ultralytics',
    '--hidden-import=torch',
    '--hidden-import=scipy',
    '--hidden-import=matplotlib',
    '--hidden-import=matplotlib.backends.backend_pdf',
    '--collect-all=ultralytics',
    '--collect-all=torch',
    '--noconfirm',                      # Sovrascrivi senza chiedere
    f'--distpath={os.path.join(current_dir, "dist")}',
    f'--workpath={os.path.join(current_dir, "build")}',
    f'--specpath={current_dir}',
])

print("\n" + "="*60)
print("✅ Build done!")
print(f"📂 You can find executable in: {os.path.join(current_dir, 'dist', 'GUV_Detector.exe')}")
print("="*60)