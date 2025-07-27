// App.tsx
import React, { useState, useEffect } from 'react';
import { ControlPanel } from './components/ControlPanel';
import { SearchGrid } from './components/SearchGrid';
import { GithubIcon } from './components/GithubIcon';
import { searchQuery, getHealth, getAllImages } from './api';
import type { SearchResult } from './api';
import { debounce } from 'lodash';
import { ImageModal } from './components/ImageModal';

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [allImages, setAllImages] = useState<string[]>([]);
  const [viewingImage, setViewingImage] = useState<string | null>(null);

  const fetchAllImages = async () => {
    try {
      const images = await getAllImages();
      setAllImages(images);
    } catch (e) {
      console.error("Failed to fetch images", e);
      setAllImages([]);
    }
  };

  const handleRebuilt = () => {
    console.log(`Index rebuilt at ${new Date().toLocaleTimeString()}`);
    setQuery('');
    setResults([]);
    fetchAllImages();
  };

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const status = await getHealth();
        if (status.loaded) {
          fetchAllImages();
        }
      } catch (e) {
        console.error("Failed to fetch initial server status", e);
      }
    };
    fetchInitialData();
  }, []);

  useEffect(() => {
    const debouncedSearch = debounce(async (searchQueryVal: string) => {
      if (!searchQueryVal) {
        setResults([]);
        setIsSearching(false);
        return;
      }
      setIsSearching(true);
      try {
        const res = await searchQuery(searchQueryVal, 12);
        setResults(res);
      } catch (error) {
        console.error("Search failed:", error);
        setResults([]);
      } finally {
        setIsSearching(false);
      }
    }, 300);

    debouncedSearch(query);

    return () => {
      debouncedSearch.cancel();
    };
  }, [query]);

  return (
    <div className="bg-gray-900 text-white min-h-screen font-sans">
      <div className="container mx-auto p-4 md:p-8">
        <header className="text-center mb-8">
          <div className="flex justify-center items-center gap-4">
            <h1 className="text-4xl md:text-5xl font-bold">Semantic Image Search</h1>
            <a href="https://github.com/coolbrg/clip-browser" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-white transition-colors">
              <GithubIcon className="w-8 h-8" />
            </a>
          </div>
          <p className="text-gray-400 mt-2">Search through your local image datasets using OpenAI's CLIP model.</p>
        </header>

        <main>
          <ControlPanel onRebuilt={handleRebuilt} query={query} setQuery={setQuery} />
          <SearchGrid results={results} isSearching={isSearching} query={query} allImages={allImages} onImageClick={setViewingImage} />
        </main>
      </div>
      <ImageModal imagePath={viewingImage} onClose={() => setViewingImage(null)} />
    </div>
  );
}

export default App
