import { useEffect, useRef, useState } from 'react';
import { TinboxAPI } from '../utils/api';
import type { ProgressUpdate } from '../types/api';

export function useProgressWebSocket(
  jobId: string | null,
  onProgress: (progress: ProgressUpdate) => void
) {
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    if (!jobId) return;

    const ws = TinboxAPI.createProgressWebSocket(
      jobId,
      (progress) => {
        setIsConnected(true);
        onProgress(progress);
      },
      () => {
        setIsConnected(false);
      }
    );

    wsRef.current = ws;

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [jobId, onProgress]);

  return { isConnected };
}
