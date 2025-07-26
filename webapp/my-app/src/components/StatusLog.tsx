// StatusLog.tsx
import React from 'react';

export const StatusLog: React.FC<{ messages: string[] }> = ({ messages }) => (
  <div className="mt-4 p-2 bg-black text-green-300 font-mono text-xs max-h-32 overflow-auto">
    {messages.map((m,i)=><div key={i}>{m}</div>)}
  </div>
);
