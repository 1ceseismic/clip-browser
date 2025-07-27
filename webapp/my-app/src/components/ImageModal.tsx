// ImageModal.tsx
import React from 'react';

interface ImageModalProps {
  imagePath: string | null;
  onClose: () => void;
}

export const ImageModal: React.FC<ImageModalProps> = ({ imagePath, onClose }) => {
  if (!imagePath) {
    return null;
  }

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div className="relative p-4" onClick={e => e.stopPropagation()}>
        <img 
          src={`http://localhost:8000/image/${encodeURIComponent(imagePath)}`} 
          alt={imagePath} 
          className="max-w-screen-lg max-h-screen-lg object-contain"
          style={{ maxHeight: '90vh', maxWidth: '90vw' }}
        />
        <button 
          onClick={onClose}
          className="absolute top-0 right-0 mt-4 mr-4 text-white bg-black bg-opacity-50 rounded-full p-2 hover:bg-opacity-75"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
};
