// SearchGrid.tsx
import React, { useState, useEffect } from 'react';
import { searchQuery } from '../api';
import type { SearchResult } from '../api';
import { debounce } from 'lodash';

export const SearchGrid: React.FC = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);

  useEffect(() => {
    const deb = debounce(async () => {
      if (!query) return setResults([]);
      const res = await searchQuery(query, 12);
      setResults(res);
    }, 200);
    deb();
    return () => deb.cancel();
  }, [query]);

  return (
    <div className="mt-4">
      <input
        className="w-full p-2 border"
        placeholder="Search…"
        value={query}
        onChange={e => setQuery(e.target.value)}
      />
      <div className="grid grid-cols-3 gap-4 mt-2">
        {results.map(r => (
          <div key={r.path} className="border rounded overflow-hidden">
            <img src={`/static/${encodeURIComponent(r.path)}`} alt="" className="w-full h-32 object-cover"/>
            <div className="p-1 text-sm">{r.score.toFixed(3)}</div>
          </div>
        ))}
      </div>
    </div>
  );
};
