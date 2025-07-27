import React, { useState, useEffect } from 'react';
import { getDirectories } from '../api';

interface DirectorySelectorProps {
  selectedDirectory: string;
  onDirectorySelect: (dir: string) => void;
}

export const DirectorySelector: React.FC<DirectorySelectorProps> = ({ selectedDirectory, onDirectorySelect }) => {
  const [directories, setDirectories] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDirs = async () => {
      try {
        const dirs = await getDirectories();
        setDirectories(dirs);
      } catch (e) {
        console.error("Failed to fetch directories", e);
        setError("Could not load directories from server.");
      } finally {
        setLoading(false);
      }
    };
    fetchDirs();
  }, []);

  if (error) {
    return <div className="text-red-500 p-2">{error}</div>;
  }

  return (
    <select 
      value={selectedDirectory} 
      onChange={e => onDirectorySelect(e.target.value)} 
      className="p-2 border rounded-lg bg-gray-700 border-gray-600 text-white h-[42px] disabled:opacity-50"
      disabled={loading}
    >
      {loading ? (
        <option>Loading...</option>
      ) : directories.length === 0 ? (
        <option value="">No datasets found</option>
      ) : (
        <>
          <option value="">Select Dataset</option>
          {directories.map(dir => (
            <option key={dir} value={dir}>
              {dir === '.' ? '(Current Folder)' : dir}
            </option>
          ))}
        </>
      )}
    </select>
  );
};
