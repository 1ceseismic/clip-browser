import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000'
});

export interface SearchResult { path: string; score: number }

export async function buildIndex(imgDir: string, checkpoint: string): Promise<{ total: number }> {
  const resp = await api.post('/build-index', { img_dir: imgDir, checkpoint });
  return resp.data;
}

export async function searchQuery(q: string, top_k = 20): Promise<SearchResult[]> {
  const resp = await api.get('/search', { params: { q, top_k } });
  return resp.data.results;
}
