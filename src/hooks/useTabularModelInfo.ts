import { useEffect, useMemo, useRef, useState } from 'react'
import type { LoadedDataset } from '../store/useDatasetStore'
import { useDatasetStore } from '../store/useDatasetStore'
import {
  computeTabularModelInfo,
  pipelineStructureKey,
  type TabularModelInfo,
} from '../editor/dataset/datasetModelInfo'
import { deferWork } from '../utils/deferWork'

const PIPELINE_PREVIEW_ROWS = 200
const DEBOUNCE_MS = 120

export function useTabularModelInfo(
  dataset: LoadedDataset,
  targetColumn: string | null,
  usePipeline: boolean,
) {
  const pipelineNodes = useDatasetStore((s) => s.pipelineNodes)
  const pipelineEdges = useDatasetStore((s) => s.pipelineEdges)

  const pipelineKey = useMemo(
    () => pipelineStructureKey(pipelineNodes, pipelineEdges),
    [pipelineNodes, pipelineEdges],
  )

  const [info, setInfo] = useState<TabularModelInfo>(() =>
    computeTabularModelInfo(dataset, targetColumn, { usePipeline: false }),
  )
  const [loading, setLoading] = useState(false)
  const requestId = useRef(0)

  useEffect(() => {
    const id = ++requestId.current

    if (!usePipeline) {
      setLoading(false)
      setInfo(computeTabularModelInfo(dataset, targetColumn, { usePipeline: false }))
      return
    }

    setLoading(true)
    const timer = setTimeout(() => {
      void deferWork().then(() => {
        if (requestId.current !== id) return
        const next = computeTabularModelInfo(dataset, targetColumn, {
          usePipeline: true,
          pipelineNodes,
          pipelineEdges,
          previewLimit: PIPELINE_PREVIEW_ROWS,
          pipelineMaxRows: PIPELINE_PREVIEW_ROWS,
        })
        if (requestId.current !== id) return
        setInfo(next)
        setLoading(false)
      })
    }, DEBOUNCE_MS)

    return () => clearTimeout(timer)
  }, [dataset, targetColumn, usePipeline, pipelineKey, pipelineNodes, pipelineEdges])

  return { info, loading }
}
