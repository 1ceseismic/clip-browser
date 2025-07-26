// ControlPanel.tsx
import React, { useState } from 'react';
import { buildIndex } from '../api';

export const ControlPanel: React.FC<{ onRebuilt: () => void }> = ({ onRebuilt }) => {
  const [dir, setDir] = useState('');
  const [checkpoint, setCheckpoint] = useState('openai');
  const [busy, setBusy] = useState(false);

  const rebuild = async () => {
    setBusy(true);
    try {
      const res = await buildIndex(dir, checkpoint);
      console.log(`Built index with ${res.total} embeddings`);
      onRebuilt();
    } catch(e) {
      console.error('Index build failed', e);
    } finally { setBusy(false); }
  };

  return (
    <div className="p-4 bg-gray-100 rounded-md flex gap-2 items-end">
      <label>Directory:
        <input type="text" value={dir} onChange={e=>setDir(e.target.value)} className="ml-1 p-1 border" />
      </label>
      <label>Model:
        <select value={checkpoint} onChange={e=>setCheckpoint(e.target.value)} className="ml-1 p-1 border">
          <option value="openai">openai</option>
          <option value="custom">custom (.pt)</option>
        </select>
      </label>
      <button disabled={busy} onClick={rebuild} className="px-3 py-1 bg-blue-600 text-white rounded disabled:opacity-50">
        {busy ? 'Building...' : 'Build Index'}
      </button>
    </div>
  );
}
