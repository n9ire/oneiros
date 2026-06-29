import type { ComponentType } from 'react'
import type { NodeProps } from '@xyflow/react'
import type { NodeData } from './graph'

export type NodeCategory = 'input' | 'output' | 'layers' | 'activation' | 'recurrent' | 'attention'

export interface NodeFieldOption {
  value: string
  label: string
}

export interface NodeField {
  key: string
  label: string
  type: 'number' | 'text' | 'select' | 'boolean'
  options?: NodeFieldOption[]
  min?: number
  max?: number
  step?: number
  placeholder?: string
}

export interface NodeDefinition {
  type: string
  label: string
  description: string
  category: NodeCategory
  defaultData: NodeData
  fields: NodeField[]
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  component: ComponentType<NodeProps<any>>
}
