// App.tsx
import React, { useState } from 'react';
import { ControlPanel } from './components/ControlPanel';
import { SearchGrid } from './components/SearchGrid';
import { StatusLog } from './components/StatusLog';

function App() {
  const [logs, setLogs] = useState<string[]>([]);

  const handleRebuilt = () => {
    setLogs(prev => [`Index rebuilt at ${new Date().toLocaleTimeString()}`, ...prev]);
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-lg font-bold">CLIP‑FAISS Semantic Browse</h1>
      <ControlPanel onRebuilt={handleRebuilt}/>
      <SearchGrid/>
      <StatusLog messages={logs}/>
    </div>
  );
}

export default App
