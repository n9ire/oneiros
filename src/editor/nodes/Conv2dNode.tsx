import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface Conv2dData extends Record<string, unknown> {
  label: string
  outChannels: number
  kernelSize: number
  stride: number
  padding: number
  activation: string
}

type Conv2dNodeType = Node<Conv2dData, 'conv2dNode'>

function Conv2dNode({ id, data, selected }: NodeProps<Conv2dNodeType>) {
  return (
    <BaseNode
      nodeId={id}
      label="Conv2d"
      category="layers"
      selected={selected}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <rect x="1" y="1" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.2" />
          <rect x="9" y="1" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
          <rect x="1" y="9" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
          <rect x="9" y="9" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.2" opacity="0.3" />
        </svg>
      }
    >
      <NodeRow label="Out Channels" value={data.outChannels} />
      <NodeRow label="Kernel" value={`${data.kernelSize}×${data.kernelSize}`} />
      <NodeRow label="Stride" value={data.stride} />
      <NodeRow label="Padding" value={data.padding} />
      <NodeRow label="Activation" value={data.activation} />
    </BaseNode>
  )
}

registerNode({
  type: 'conv2dNode',
  label: 'Conv2d',
  description: '2D convolution layer',
  category: 'layers',
  defaultData: {
    label: 'Conv2d',
    outChannels: 32,
    kernelSize: 3,
    stride: 1,
    padding: 0,
    activation: 'relu',
  },
  fields: [
    { key: 'outChannels', label: 'Out Channels', type: 'number', min: 1 },
    { key: 'kernelSize', label: 'Kernel Size', type: 'number', min: 1 },
    { key: 'stride', label: 'Stride', type: 'number', min: 1 },
    { key: 'padding', label: 'Padding', type: 'number', min: 0 },
    {
      key: 'activation',
      label: 'Activation',
      type: 'select',
      options: [
        { value: 'relu', label: 'ReLU' },
        { value: 'leaky_relu', label: 'Leaky ReLU' },
        { value: 'sigmoid', label: 'Sigmoid' },
        { value: 'tanh', label: 'Tanh' },
        { value: 'none', label: 'None' },
      ],
    },
  ],
  component: Conv2dNode,
})

export default Conv2dNode
