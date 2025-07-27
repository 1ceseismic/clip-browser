// SearchGrid.tsx
import React from 'react';
import type { SearchResult } from '../api';

interface SearchGridProps {
  results: SearchResult[];
  isSearching: boolean;
  query: string;
  allImages: string[];
  onImageClick: (path: string) => void;
}

export const SearchGrid: React.FC<SearchGridProps> = ({ results, isSearching, query, allImages, onImageClick }) => {
  if (isSearching) {
    return <div className="text-center text-gray-400 pt-8">Searching...</div>;
  }

  // In search mode (query is not empty)
  if (query) {
    if (results.length === 0) {
      return <div className="text-center text-gray-400 pt-8">No results found for "{query}".</div>;
    }
    return (
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
        {results.map(r => (
          <div key={r.path} onClick={() => onImageClick(r.path)} className="group relative border-2 border-gray-800 rounded-lg overflow-hidden hover:border-blue-500 transition-all duration-300 cursor-pointer">
            <img src={`http://localhost:8000/thumbnail/${encodeURIComponent(r.path)}`} alt={r.path} className="w-full h-40 object-cover group-hover:opacity-75 transition-opacity"/>
            <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 p-1 text-xs text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity">
              Score: {r.score.toFixed(3)}
            </div>
          </div>
        ))}
      </div>
    );
  }

  // In browse mode (no query)
  if (allImages.length === 0) {
    return <div className="text-center text-gray-400 pt-8">Build an index to see your images.</div>;
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
      {allImages.map(path => (
        <div key={path} onClick={() => onImageClick(path)} className="group relative border-2 border-gray-800 rounded-lg overflow-hidden hover:border-blue-500 transition-all duration-300 cursor-pointer">
          <img src={`http://localhost:8000/thumbnail/${encodeURIComponent(path)}`} alt={path} className="w-full h-40 object-cover"/>
        </div>
      ))}
    </div>
  );
};
