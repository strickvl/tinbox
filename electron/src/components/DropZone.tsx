import { useState, useCallback } from 'react';
import { Upload } from './Icons';

interface DropZoneProps {
  onFilesSelected: (files: string[]) => void;
  disabled?: boolean;
}

export function DropZone({ onFilesSelected, disabled }: DropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) {
      setIsDragging(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      if (disabled) return;

      const files = Array.from(e.dataTransfer.files);
      const validExtensions = ['.pdf', '.docx', '.txt'];
      const validFiles = files.filter((file) =>
        validExtensions.some((ext) => file.name.toLowerCase().endsWith(ext))
      );

      if (validFiles.length > 0) {
        onFilesSelected(validFiles.map((f) => f.path));
      }
    },
    [onFilesSelected, disabled]
  );

  const handleClick = async () => {
    if (disabled) return;

    const paths = await window.electron.openFileDialog({
      filters: [
        { name: 'Documents', extensions: ['pdf', 'docx', 'txt'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });

    if (paths.length > 0) {
      onFilesSelected(paths);
    }
  };

  return (
    <div
      className={`
        relative flex flex-col items-center justify-center
        border-2 border-dashed rounded-lg p-12 transition-all
        ${isDragging ? 'border-primary-500 bg-primary-50' : 'border-gray-300 bg-white'}
        ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:border-primary-400 hover:bg-gray-50'}
      `}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
    >
      <Upload className={`w-16 h-16 mb-4 ${isDragging ? 'text-primary-500' : 'text-gray-400'}`} />

      <h3 className="text-lg font-medium text-gray-900 mb-2">
        {isDragging ? 'Drop files here' : 'Drop files to translate'}
      </h3>

      <p className="text-sm text-gray-500 mb-4">
        or click to browse
      </p>

      <div className="flex gap-2 text-xs text-gray-400">
        <span className="px-2 py-1 bg-gray-100 rounded">PDF</span>
        <span className="px-2 py-1 bg-gray-100 rounded">DOCX</span>
        <span className="px-2 py-1 bg-gray-100 rounded">TXT</span>
      </div>
    </div>
  );
}
