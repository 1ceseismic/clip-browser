import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000'
});

export interface SearchResult { path: string; score: number }
export interface HealthStatus { loaded: boolean; indexed_dir: string | null; status: string; }

export async function setDatasetRoot(path: string): Promise<{ status: string, path: string }> {
  const resp = await api.post('/set-dataset-root', { path });
  return resp.data;
}

export async function buildIndex(imgDir: string, checkpoint: string): Promise<{ total: number }> {
  // The backend endpoint expects these as query parameters, not a JSON body.
  const resp = await api.post('/build-index', null, { params: { img_dir: imgDir, checkpoint } });
  return resp.data;
}

export async function searchQuery(q: string, top_k = 20): Promise<SearchResult[]> {
  const resp = await api.get('/search', { params: { q, top_k } });
  return resp.data.results;
}

export async function getDirectories(): Promise<string[]> {
  // This new endpoint will be created in app.py
  const resp = await api.get('/directories');
  return resp.data.directories;
}

export async function getAllImages(): Promise<string[]> {
  const resp = await api.get('/all-images');
  return resp.data.images;
}

export async function getHealth(): Promise<HealthStatus> {
  const resp = await api.get('/health');
  return resp.data;
}
