// ControlPanel.tsx
import React, { useState, useEffect } from 'react';
import { buildIndex, getHealth, setDatasetRoot } from '../api';
import { DirectorySelector } from './DirectorySelector';

interface ControlPanelProps {
  onRebuilt: () => void;
  query: string;
  setQuery: (query: string) => void;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({ onRebuilt, query, setQuery }) => {
  const [rootPath, setRootPath] = useState('');
  const [isRootLoading, setIsRootLoading] = useState(false);
  const [isRootSet, setIsRootSet] = useState(false);
  const [rootError, setRootError] = useState<string | null>(null);
  
  const [dir, setDir] = useState('');
  const [checkpoint, setCheckpoint] = useState('openai');
  const [isBuilding, setIsBuilding] = useState(false);
  const [isIndexLoaded, setIsIndexLoaded] = useState(false);
  const [dirSelectorToken, setDirSelectorToken] = useState(0);

  useEffect(() => {
    const checkServerStatus = async () => {
      try {
        const status = await getHealth();
        setIsIndexLoaded(status.loaded);
        if (status.loaded && status.indexed_dir) {
          setDir(status.indexed_dir);
        }
      } catch (e) {
        console.error("Could not contact server for health check.", e);
      }
    };
    checkServerStatus();
  }, []);

  const handleSetRootPath = async () => {
    if (!rootPath) return;
    setIsRootLoading(true);
    setRootError(null);
    try {
      await setDatasetRoot(rootPath);
      setIsRootSet(true);
      setDirSelectorToken(t => t + 1); // Force DirectorySelector to re-fetch
    } catch (e) {
      console.error("Failed to set root path", e);
      setRootError("Failed to load path. Check if it's a valid directory path on the server.");
      setIsRootSet(false);
    } finally {
      setIsRootLoading(false);
    }
  };

  const handleBuildIndex = async () => {
    if (!dir) {
      alert("Please select a dataset to index.");
      return;
    }
    setIsBuilding(true);
    try {
      const res = await buildIndex(dir, checkpoint);
      console.log(`Built index with ${res.total} embeddings`);
      setIsIndexLoaded(true);
      onRebuilt();
    } catch(e) {
      console.error('Index build failed', e);
      alert('Failed to build index. Check the server logs for more details.');
    } finally {
      setIsBuilding(false);
    }
  };

  return (
    <div className="p-4 bg-gray-800 rounded-lg shadow-lg mb-8 space-y-4">
      {/* --- Step 1: Set Root Path --- */}
      <div>
        <label htmlFor="root-path-input" className="block text-sm font-medium text-gray-300 mb-1">
          Dataset Root Path
        </label>
        <div className="flex gap-2">
          <input
            id="root-path-input"
            type="text"
            placeholder="e.g., C:\Users\YourName\Pictures or /home/user/images"
            value={rootPath}
            onChange={e => setRootPath(e.target.value)}
            className="flex-grow p-2 border rounded-lg bg-gray-700 border-gray-600 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isRootSet}
          />
          {isRootSet ? (
            <button onClick={() => setIsRootSet(false)} className="px-4 py-2 bg-yellow-600 text-white font-semibold rounded-lg hover:bg-yellow-700 transition-colors h-[42px]">
              Change
            </button>
          ) : (
            <button onClick={handleSetRootPath} disabled={isRootLoading || !rootPath} className="px-4 py-2 bg-green-600 text-white font-semibold rounded-lg disabled:opacity-50 hover:bg-green-700 transition-colors h-[42px]">
              {isRootLoading ? 'Loading...' : 'Load'}
            </button>
          )}
        </div>
        {rootError && <p className="text-red-400 text-sm mt-1">{rootError}</p>}
      </div>

      {/* --- Step 2: Index and Search (visible only after root is set) --- */}
      {isRootSet && (
        <div className="border-t border-gray-700 pt-4">
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <DirectorySelector key={dirSelectorToken} selectedDirectory={dir} onDirectorySelect={setDir} />
              <select value={checkpoint} onChange={e=>setCheckpoint(e.target.value)} className="p-2 border rounded-lg bg-gray-700 border-gray-600 text-white h-[42px]">
                <option value="openai">OpenAI</option>
                <option value="custom">Custom</option>
              </select>
              <button disabled={isBuilding || !dir} onClick={handleBuildIndex} className="px-4 py-2 bg-blue-600 text-white font-semibold rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors h-[42px] whitespace-nowrap">
                {isBuilding ? 'Building...' : (isIndexLoaded ? 'Rebuild Index' : 'Build Index')}
              </button>
            </div>
            <div className="flex-grow hidden sm:block"></div>
            <div className="relative w-full sm:w-auto sm:flex-grow">
              <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                type="text"
                className="w-full p-2 pl-10 border rounded-lg bg-gray-700 border-gray-600 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                placeholder="Search for images..."
                value={query}
                onChange={e => setQuery(e.target.value)}
                disabled={!isIndexLoaded || isBuilding}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
