from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import plotly.graph_objs as go
import numpy as np
import os
from prody import parsePDB, parseDCD, Ensemble
import gc
import shutil
import tempfile

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ProDy tabanlı analiz fonksiyonu
def prody_rmsd_rmsf(pdb_path, dcd_path, chunk_size=250, total_frames=None):
    structure = parsePDB(pdb_path)
    structure_calpha = structure.select("calpha")
    if structure_calpha is None:
        return None, None, None, None, "PDB dosyasında Cα atomu bulunamadı!"
    residue_number = len(structure_calpha)
    if total_frames is None:
        from prody import DCDFile
        with DCDFile(dcd_path) as dcd:
            total_frames = len(dcd)
    rmsd_all = []
    all_coords = []
    for start in range(0, total_frames, chunk_size):
        stop = min(start + chunk_size, total_frames)
        ensemble_cov = parseDCD(dcd_path, start=start, stop=stop)
        ensemble_cov.setAtoms(structure_calpha)
        ensemble_cov.setCoords(structure)
        ensemble_cov.superpose()
        rmsd = ensemble_cov.getRMSDs()
        rmsd_all.extend(rmsd)
        all_coords.append(ensemble_cov.getCoordsets())
        gc.collect()
    # Tüm frame'leri birleştir
    all_coords = np.concatenate(all_coords, axis=0)
    # ProDy Ensemble ile RMSF hesapla
    ens = Ensemble('all')
    ens.setCoords(structure_calpha.getCoords())
    ens.addCoordset(all_coords)
    rmsf_all = ens.getRMSFs()
    res_ids = np.arange(1, len(rmsf_all)+1)
    return np.array(rmsd_all), rmsf_all, res_ids, res_ids, None

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "rmsd_plot": None, "rmsf_plot": None, "error": None})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, pdb_file: UploadFile = File(...), dcd_file: UploadFile = File(...)):
    error = None
    rmsd_plot = None
    rmsf_plot = None
    try:
        # Dosyaları geçici dizine kaydet
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdb') as tmp_pdb:
            shutil.copyfileobj(pdb_file.file, tmp_pdb)
            pdb_path = tmp_pdb.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcd') as tmp_dcd:
            shutil.copyfileobj(dcd_file.file, tmp_dcd)
            dcd_path = tmp_dcd.name
        # ProDy ile analiz
        rmsd, rmsf, rmsf_idx, res_ids, error = prody_rmsd_rmsf(pdb_path, dcd_path)
        if error:
            return templates.TemplateResponse("index.html", {"request": request, "rmsd_plot": None, "rmsf_plot": None, "error": error})
        # RMSD plot
        fig_rmsd = go.Figure()
        fig_rmsd.add_trace(go.Scatter(x=np.arange(1, len(rmsd)+1), y=rmsd, mode='lines', name='RMSD'))
        fig_rmsd.update_layout(title='RMSD (Cα, ProDy)', xaxis_title='Frame', yaxis_title='RMSD (Å)')
        rmsd_plot = fig_rmsd.to_html(full_html=False)
        # RMSF plot (y eksenini Å olarak göster)
        fig_rmsf = go.Figure()
        fig_rmsf.add_trace(go.Scatter(x=res_ids, y=rmsf, mode='lines+markers', name='RMSF (Cα)'))
        fig_rmsf.update_layout(title='RMSF Profili (Cα, ProDy)', xaxis_title='Residue ID', yaxis_title='RMSF (Å)')
        rmsf_plot = fig_rmsf.to_html(full_html=False)
        # Geçici dosyaları sil
        try:
            os.remove(pdb_path)
            os.remove(dcd_path)
        except Exception:
            pass
    except Exception as e:
        error = f"Analiz hatası: {e}"
    return templates.TemplateResponse("index.html", {"request": request, "rmsd_plot": rmsd_plot, "rmsf_plot": rmsf_plot, "error": error}) 