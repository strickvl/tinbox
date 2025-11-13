import { useState } from 'react';
import { TinboxAPI } from '../utils/api';
import { useStore } from '../store/useStore';
import type { TranslateRequest, CostEstimate } from '../types/api';

export function useTranslation() {
  const [isEstimating, setIsEstimating] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const addJob = useStore((state) => state.addJob);

  const estimateCost = async (
    filePath: string,
    model: string
  ): Promise<CostEstimate | null> => {
    setIsEstimating(true);
    try {
      const estimate = await TinboxAPI.estimateCost(filePath, model);
      return estimate;
    } catch (error) {
      console.error('Failed to estimate cost:', error);
      return null;
    } finally {
      setIsEstimating(false);
    }
  };

  const startTranslation = async (
    request: TranslateRequest
  ): Promise<string | null> => {
    setIsStarting(true);
    try {
      const job = await TinboxAPI.startTranslation(request);
      addJob(job);
      return job.job_id;
    } catch (error) {
      console.error('Failed to start translation:', error);
      alert(`Failed to start translation: ${error}`);
      return null;
    } finally {
      setIsStarting(false);
    }
  };

  return {
    estimateCost,
    startTranslation,
    isEstimating,
    isStarting,
  };
}
