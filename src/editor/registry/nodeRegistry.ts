import type { ComponentType } from 'react'
import type { NodeTypes } from '@xyflow/react'
import type { NodeDefinition, NodeCategory } from '../../types/node'

const registry = new Map<string, NodeDefinition>()

export function registerNode(definition: NodeDefinition): void {
  registry.set(definition.type, definition)
}

export function getNodeDefinition(type: string): NodeDefinition | undefined {
  return registry.get(type)
}

export function getAllNodes(): NodeDefinition[] {
  return [...registry.values()]
}

export function getNodesByCategory(category: NodeCategory): NodeDefinition[] {
  return [...registry.values()].filter((def) => def.category === category)
}

export function getXYFlowNodeTypes(): NodeTypes {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return Object.fromEntries(
    [...registry.entries()].map(([type, def]) => [type, def.component as ComponentType<any>])
  ) as NodeTypes
}
